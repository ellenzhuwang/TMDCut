"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse
import random
import pickle

import torch
import datetime
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou, bbox_iou_multi
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
from object_discovery import ncut 
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    parser.add_argument("--box_path", type = str, default=None, help="Path for predicted bounding box" )

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    os.makedirs(args.output_dir, exist_ok=True)

    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    start_time = time.time() 
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        
        # print(sd)
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue
        
        box_file_name = im_name.replace('.jpg', '.npy')

        box_file_path = os.path.join(args.box_path, box_file_name)

        if not os.path.exists(box_file_path):
            continue
        
        pred = np.load(box_file_path)
        # print(pred)

        

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue
        
        # print("gt_bbxs",gt_bbxs)

        # print(df)

        # Compare prediction to GT boxes
        ious = bbox_iou_multi(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
        
        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    end_time = time.time()
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')

    # Evaluate
    folder = f"{args.output_dir}"
    print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
    result_file = os.path.join(folder, 'results.txt')
    with open(result_file, 'w') as f:
        f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
    print('File saved at %s'%result_file)
