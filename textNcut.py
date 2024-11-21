import torch
import torch.nn.functional as F
import numpy as np
from myutils import MaskPooling

class TMDCut:
    def get_affinity_matrix(self, affinity, level):
        size = affinity.shape[0]
        null_values_idx = set(torch.unique(torch.where(affinity==0)[0]).tolist())
        all_idx = set(np.arange(0, size, 1))
        if level == 0:
            idx2keep = list(all_idx)
            A = affinity
        else:
            idx2keep = list(all_idx - null_values_idx)
            A = affinity[:, idx2keep][idx2keep, :]
        D = self.get_degree_matrix(A)
        return A, D, idx2keep

    def get_degree_matrix(self, A):
        d_i = torch.sum(A, axis=1)
        D = torch.diag(d_i)
        return D

    def second_smallest_eigenvector(self, A, D):
        # get the second smallest eigenvector from affinity matrix
        D_inv_sqrt = torch.diag(1 / torch.diag(D) ** 0.5)
        try:
            _, eigenvectors = torch.linalg.eigh(D_inv_sqrt @ (D - A) @ D_inv_sqrt)
            second_smallest_vec = D_inv_sqrt @ eigenvectors[:, 1]
        except torch._C._LinAlgError:
            second_smallest_vec = None
        return second_smallest_vec

    def get_bipartition(self, y, A, D, k=100):
        xmin, xmax = y.min() * 0.99, y.max() * 0.99
        thresholds = torch.linspace(xmin, xmax, k)
        partitions = [1. * (y > thresh) for thresh in thresholds]
        N_cut_values = []
        L = D - A
        sum_D = torch.sum(D)
        diag_D = torch.diag(D)

        for x in partitions:
            x = 2 * x - 1
            k = torch.sum(diag_D * (x > 0)) / sum_D
            b = k / (1 - k)
            y = (1 + x) - b * (1 - x)
            E = (torch.t(y) @ L @ y) / (torch.t(y) @ D @ y)
            N_cut_values.append(E.item())

        ncut = min(N_cut_values)
        bipartition = partitions[np.argmin(N_cut_values)]
        return bipartition, ncut

    def get_masked_affinity_matrix(self, painting, affinity, mask):
        # mask out affinity matrix based on the painting matrix
        painting = painting + mask.unsqueeze(0)
        painting[painting > 0] = 1
        painting[painting <= 0] = 0
        mask_affinity = (torch.t(painting.reshape(1, -1)) @ painting.reshape(1, -1))
        affinity = affinity * (1 - mask_affinity)
        return affinity, painting

    def deterministic_vector_sign_flip(self, u):
        max_abs_idx = torch.argmax(torch.abs(u))
        sign = torch.sign(u[max_abs_idx])
        u *= sign
        return u

    
    def recursive_ncut(self, feats, max_cuts,dims=(64, 64), painting=None, mask=None, accumulator_masks=None, level=0,
    cuts_made=None):

        if cuts_made is None:
            cuts_made = [0]  # Initialize the counter as a list for mutability across recursive calls

        if level == 0:
            accumulator_masks = []
            feats = F.normalize(feats, p=2, dim=0)
            painting = torch.zeros(dims).to("cuda")
            mask = torch.zeros(dims).to("cuda")
            affinity = torch.t(feats) @ feats
            affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min())
            affinity = affinity ** alpha

        if level > 0:
            affinity = feats

            np.save('affinity_matrix.npy', affinity.to('cpu'))

        # Mask the explored area in the affinity matrix
        affinity, painting = self.get_masked_affinity_matrix(painting, affinity, mask)
        #print(affinity.shape)
        #print(painting.shape)
        #print(mask.shape)

        # Construct a smaller affinity matrix (A) and degree matrix (D)
        A, D, idx2keep = self.get_affinity_matrix(affinity, level)

        if A.shape[0] > 1 and cuts_made[0] < max_cuts:
            # get the second smallest eigenvector
            second_smallest_vec = self.second_smallest_eigenvector(A, D)
            second_smallest_vec = self.deterministic_vector_sign_flip(second_smallest_vec)
            # Map back to the full image size
            full_vec = torch.zeros((1, dims[0] * dims[1])).to("cuda")
            full_vec[:, idx2keep] = second_smallest_vec

            # Create bipartition mask based on the sign of the eigenvector values
            bipartition = (full_vec >= 0).float().reshape(dims)

            # Increment the cuts_made counter
            cuts_made[0] += 1

            # Update the accumulator_masks with the new bipartition
            accumulator_masks.append(bipartition)

            # Recursively process both partitions
            self.recursive_ncut(
                affinity,
                dims=dims,
                painting=painting.to('cuda'),
                mask=1 - bipartition,
                accumulator_masks=accumulator_masks,
                level=level + 1,
                max_cuts=max_cuts,
                cuts_made=cuts_made
            )
            self.recursive_ncut(
                affinity,
                dims=dims,
                painting=painting.to("cuda"),
                mask=bipartition,
                accumulator_masks=accumulator_masks,
                level=level + 1,
                max_cuts=max_cuts,
                cuts_made=cuts_made
            )
        else:
            # If no further cuts can be made, append the current mask
            if mask is not None and mask.sum() > 0:
                accumulator_masks.append(mask)

        return accumulator_masks

    def assemble_clusters(self, clusters, h, w):
        mask = torch.zeros((h, w)).to("cuda")
        final_mask = torch.zeros((h, w)).to("cuda")
        max_value = 1
        for cluster_mask in clusters:
            mask += max_value * cluster_mask
            max_value = mask.max() + 1

        for i, cls_idx in enumerate(torch.unique(mask)):
            final_mask[mask==cls_idx] = i

        return final_mask

    def generate_masks(self, features, mask_size=(32, 32), masks=None, img_size=2048):

        with open('./voc12_n_objects.json', 'r') as file:
            data = json.load(file)

        max_cuts = data[file_name[0]]
        if max_cuts < 20:
            max_cuts = max_cuts

        else:
            max_cuts = 20

        h_mask, w_mask = mask_size
        mask_pooling = MaskPooling()
        
        feats = features.reshape(1, img_size//64, img_size//64, -1).permute(0, 3, 1, 2).to(torch.float32)
        feats_norm = F.normalize(feats, dim=1)
        _, c, h, w = feats_norm.shape

        if masks is None:
            x = feats_norm.reshape(c, h*w)
            clusters = self.recursive_ncut(x, max_cuts=max_cuts,dims=(h, h))
            #clusters = self.recursive_ncut(x,tau=tau, dims=(h, h), alpha=alpha)

            if len(clusters) == 0:
                clusters.append(torch.zeros((h, w)).to("cuda"))

            # Assemble clusters
            masks = self.assemble_clusters(clusters, h, w)[None, None]

        clusters_embeddings = torch.zeros(len(torch.unique(masks)), c).to("cuda")

        for k, cls_idx in enumerate(torch.unique(masks)):
            mask = masks == cls_idx
            mask_embed = mask_pooling(feats_norm, mask)[0]
            clusters_embeddings[k, :] = mask_embed

        upsampled_feats = torch.nn.Upsample(size=mask_size, mode='bilinear')(feats_norm)


        similarity_scores = torch.t(upsampled_feats.reshape(c, h_mask*w_mask)) @ torch.t(clusters_embeddings)
        masks = torch.argmax(similarity_scores, dim=1).reshape(1, 1, h_mask, w_mask)

        masks = masks.cpu().numpy().astype(int)

        # masks = F.softmax(similarity_scores, dim=1).reshape(1, -1, h_mask, w_mask)
        # Obtain soft masks from similarity scores
        # similarity_scores = torch.t(upsampled_feats.reshape(c, h_mask * w_mask)) @ clusters_embeddings.t()
        # soft_masks = F.softmax(similarity_scores, dim=1).reshape(1, -1, h_mask, w_mask)
        # resized_soft_masks = F.interpolate(soft_masks, size=crop_seg_logit.shape[2:], mode='bilinear', align_corners=False)
        # crop_seg_logit = crop_seg_logit * resized_soft_masks
        return masks
