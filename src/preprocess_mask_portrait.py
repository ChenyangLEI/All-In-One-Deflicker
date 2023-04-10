import numpy as np
# set gpu
import os
import subprocess

import argparse
from pathlib import Path
import cv2

import sys
sys.path.append("./image-background-remove-tool")
from carvekit.api.high import HiInterface
import matplotlib.image as mpimg
import torch

def preprocess(args):
    images = sorted(args.vid_path.glob('*.png'))
    vid_name = args.vid_path.name
    vid_root = args.vid_path.parent
    out_mask_dir = vid_root / f'{vid_name}_seg'
    out_mask_dir.mkdir(exist_ok=True)


    # Check doc strings for more information
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)    
    
    number_of_frames = len(images)

    for i in range(number_of_frames):
        images_without_background = interface([images[i]])
        cat_wo_bg = images_without_background[0]
        mask = np.array(cat_wo_bg)[:, :, 3]
        cv2.imwrite(f"{out_mask_dir}/%05d.png"%(i),  mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument(
        '--vid-path', type=Path, default=Path('./data/'), help='folder to process')
    parser.add_argument('--gpu', type=int,default=0, help='gpu id')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    
    preprocess(args=args)
