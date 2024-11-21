import torch
import torch.nn as nn
import sys
import math
import numpy as np

from scipy.ndimage import median_filter

sys.path.append("..")

from prompts.imagenet_template import openai_imagenet_template, sub_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange
import random

from open_clip import create_model, tokenizer, transformer
from segment_anything import sam_model_registry

from myutils import UnNormalize
from collections import namedtuple

from textNcut import TMDCut

Patch = namedtuple('Patch', ['x1', 'y1', 'x2', 'y2'])

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

import torch
import torch.nn as nn

class TMDLayer(nn.Module):
    def __init__(
        self,
        in_features=768,  # Updated to match ViT's feature dimension
        L_latent=16,
        epsilon=0.3
    ):
        super().__init__()

        # Define layers for projecting features and pi_list
        self.pi_list = nn.Sequential(
            nn.Linear(L_latent, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        self.dt = nn.Parameter(torch.FloatTensor([0.05]))
        self.epsilon = epsilon

        # Projection layer to reduce feature dimension to latent space
        #self.proj_list = nn.Sequential(nn.Linear(in_features, L_latent))

        self.proj_list = nn.Sequential(
            nn.Linear(in_features, L_latent),
            nn.LayerNorm(L_latent),  # Adding LayerNorm to the projected latent features
            nn.ReLU()
        )

        self.l_norm = nn.LayerNorm(in_features)

    def TMD_map(self, x):
        # Ensure the input tensor `x` matches the data type of the model parameters
        x = x.to(self.proj_list[0].weight.dtype)

        # Proceed with the rest of the TMD_map logic
        B, N, d = x.shape

        # Project features to latent space
        x_proj = self.proj_list(x)  # [B, N, L_latent]

        # Construct pairwise distances
        i_minus_j = x_proj.unsqueeze(2) - x_proj.unsqueeze(1)  # [B, N, N, L_latent]
        K_epsilon = torch.exp(-1 / (4 * self.epsilon) * (i_minus_j ** 2).sum(dim=3))  # [B, N, N]

        # Construct TMD
        q_epsilon_tilde = K_epsilon.sum(dim=2)  # [B, N]
        pi_values = self.pi_list(x_proj).squeeze(2)  # [B, N]
        D_epsilon_tilde = torch.diag_embed(pi_values / q_epsilon_tilde)  # [B, N, N]
        K_tilde = K_epsilon.bmm(D_epsilon_tilde)  # [B, N, N]
        D_tilde = torch.diag_embed(K_tilde.sum(dim=2) + 1e-5)  # [B, N, N]

        L = (1 / self.epsilon) * (torch.inverse(D_tilde).bmm(K_tilde)) - torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return L

    def forward(self, x, f):
        # Get L matrix
        L = self.TMD_map(x).squeeze(0)  # [B, N, N]

        print(L.shape)

        # Get target using function f
        target = f(x)  # Assuming f returns features of shape [B, N, d]
        print(target.shape)

        # Ensure all tensors are of the same data type
        target = target.to(L.dtype)
        self.dt = self.dt.to(L.dtype)

        # Introduce a learnable scaling factor
        alpha = 0.7  # You can also make this a learnable parameter for more flexibility
        target_transformed = self.dt * torch.matmul(L, target)

        # Apply transformation
        target = target + self.dt * torch.matmul(L, target)  # [B, N, d]
        # Use a residual connection and scale the influence of TMD
        target = alpha * x + (1 - alpha) * (target + target_transformed)
        target = self.l_norm(target)
        return target

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.to(self.qkv.weight.dtype)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@MODELS.register_module()
class TMDCut_Segmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, vfm_model, name_path, checkpoint=None, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)
        self.attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8).half()

        self.ncut = TMDCut()

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.attention = transformer.Attention
        self.residual = transformer.ResidualAttentionBlock

        self.vfm_model = vfm_model
        if vfm_model == 'sam':
            self.vfm = sam_model_registry["vit_b"](checkpoint=checkpoint)
            # self.vfm = sam_model_registry["vit_l"](checkpoint=checkpoint)

        elif vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif vfm_model == 'dinov2':
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        else:
            print("vlm_model not supported")

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                #print(qw)
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

    @torch.no_grad()
    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)

        imgs_norm = imgs_norm.half()

        if self.vfm_model == 'sam':
            patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size
            imgs_norm = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            img_feats = self.vfm.image_encoder(imgs_norm)

        elif self.vfm_model == 'dino':
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tmd_layer = TMDLayer(in_features=768).to(device)
            feature_transform_fn = Attention(dim=768).to(device)
            feat = tmd_layer(feat, feature_transform_fn)

            nb_im = feat.shape[0]  # Batch size
            nb_tokens = feat.shape[1]  # Number of tokens
            nh = self.vfm.blocks[0].attn.num_heads  # Number of heads
            feature_dim = feat.shape[2]

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]

            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size
            img_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)

        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            img_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]

        else:
            I, J = clip_token_size
            img_feats = None

        image_features = self.clip.encode_image(img.half(),beta=self.beta,gamma=self.gamma)
        
        tmd_layer2 = TMDLayer(in_features=512).to(device)
        feature_transform_fn = Attention(dim=768).to(device)
        image_features = tmd_layer(image_features, feature_transform_fn)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)

        if logit_size == None:
            #logits = nn.functional.interpolate(logits_map, size=img.shape[-2:], mode='bilinear', align_corners=False)

            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')


        masks = self.ncut.generate_masks(image_features)

        masks = torch.Tensor(masks).to("cuda")
        im =  F.interpolate(imgs_norm, size=(64,64), mode='bilinear')
        #crop_mask = self.pamr(crop_mask, crop_img)[None]
        
        
        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks, device=crop_seg_logit.device, dtype=torch.float32)

        resized_masks = F.interpolate(masks, size=(crop_seg_logit.shape[2], crop_seg_logit.shape[3]), mode='bilinear', align_corners=False)
        resized_masks = resized_masks.expand(-1, crop_seg_logit.shape[1], -1, -1)

        logits = logits * resized_masks
        torch.cuda.empty_cache()

        # mask cutting for padded image
        if any(pad):
            l, t = pad[0], pad[2]
            logits = logits[:, :, t:t + H, l:l + W]

        preds += nn.functional.pad(logits,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))
        
        count_mat[:, :, y1:y2, x1:x2] += 1
        
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return image_features,logits

    def predict(self, inputs, data_samples):
        #print(inputs)
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        
        seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(inputs,seg_logits, data_samples)

    def postprocess_result(self, img, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        #print(seg_logits.shape)
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h
            #print("seg_logits",seg_logits.shape)

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                #print(seg_logits.shape)
                seg_logits = (seg_logits * cls_index).max(1)[0]
            
            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            #seg_pred = torch.Tensor(seg_pred).to("cuda")
            #im =  F.interpolate(img, size=(seg_pred.shape[1], seg_pred.shape[2]), mode='bilinear')
            #seg_pred = self.pamr(seg_pred, im)[None]

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices