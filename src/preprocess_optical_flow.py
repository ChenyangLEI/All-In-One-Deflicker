import numpy as np
# set gpu
import os
from pathlib import Path
import argparse
from src.models.stage_1.raft_wrapper import RAFTWrapper

from tqdm import tqdm

def preprocess(args):
    files = sorted(args.vid_path.glob('*.*g'))
    vid_name = args.vid_path.name
    vid_root = args.vid_path.parent
    out_flow_dir = vid_root / f'{vid_name}_flow'
    out_flow_dir.mkdir(exist_ok=True)
    raft_wrapper = RAFTWrapper(
        model_path='pretrained_weights/raft-things.pth', max_long_edge=args.max_long_edge)
    for i, file1 in enumerate(tqdm(files,desc='computing flow')):
        if i < len(files) - 1:
            file2 = files[i + 1]
            fn1 = file1.name
            fn2 = file2.name
            out_flow12_fn = out_flow_dir / f'{fn1}_{fn2}.npy'
            out_flow21_fn = out_flow_dir / f'{fn2}_{fn1}.npy'

            overwrite=False
            if not out_flow12_fn.exists() and not out_flow21_fn.exists() or overwrite:
                im1, im2 = raft_wrapper.load_images(str(file1), str(file2))
                flow12 = raft_wrapper.compute_flow(im1, im2)
                flow21 = raft_wrapper.compute_flow(im2, im1)
                np.save(out_flow12_fn, flow12)
                np.save(out_flow21_fn, flow21)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument(
        '--vid-path', type=Path, default=Path('./data/'), help='folder to process')
    # Need to reduce when there are memory issues
    parser.add_argument('--max_long_edge', type=int,default='2000', help='maximum image dimension to process without resizing')
    parser.add_argument('--gpu', type=int,default=0, help='gpu id')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    preprocess(args=args)
