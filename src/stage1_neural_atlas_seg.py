import sys
import torch
import torch.optim as optim
import numpy as np
import argparse
import cv2
import glob
from tqdm import tqdm

from src.models.stage_1.implicit_neural_networks import IMLP
from src.models.stage_1.evaluate import evaluate_model
from src.models.stage_1.loss_utils import get_gradient_loss, get_rigidity_loss, get_optical_flow_loss, get_optical_flow_alpha_loss
from src.models.stage_1.unwrap_utils import get_tuples, pre_train_mapping, load_input_data, save_mask_flow

import json
from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# set gpu
import os
import subprocess

def main(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum_number_of_frames = config["maximum_number_of_frames"]
    # read the first frame of vid path and get its resolution
    frames_list = sorted(glob.glob(os.path.join(args.vid_path, "*g")))
    frame_temp = cv2.imread(frames_list[0])
    resx = frame_temp.shape[1]
    resy = frame_temp.shape[0]
    
    if args.down is not None:
        resx = int(resx / args.down)
        resy = int(resy / args.down)
        
    iters_num = config["iters_num"]

    #batch size:
    samples = config["samples_batch"]

    # evaluation frequency (in terms of iterations number)
    evaluate_every = np.int64(config["evaluate_every"])

    # optionally it is possible to load a checkpoint
    load_checkpoint = config["load_checkpoint"] # set to true to continue from a checkpoint
    checkpoint_path = config["checkpoint_path"]

    # a data folder that contains folders named "[video_name]","[video_name]_flow","[video_name]_maskrcnn"
    data_folder = Path(args.vid_path)
    results_folder_name = "results"

    # boolean variables for determining if a pretraining is used:
    pretrain_mapping1 = config["pretrain_mapping1"]
    pretrain_mapping2 = config["pretrain_mapping2"]
    pretrain_iter_number = config["pretrain_iter_number"]

    # the scale of the atlas uv coordinates relative to frame's xy coordinates
    uv_mapping_scale = config["uv_mapping_scale"]

    # M_\alpha's hyper parameters:
    positional_encoding_num_alpha = config["positional_encoding_num_alpha"]
    number_of_channels_alpha = config["number_of_channels_alpha"]
    number_of_layers_alpha = config["number_of_layers_alpha"]

    # M_f's hyper parameters
    use_positional_encoding_mapping1 = config["use_positional_encoding_mapping1"]
    number_of_positional_encoding_mapping1 = config["number_of_positional_encoding_mapping1"]
    number_of_layers_mapping1 = config["number_of_layers_mapping1"]
    number_of_channels_mapping1 = config["number_of_channels_mapping1"]

    # M_b's hyper parameters
    use_positional_encoding_mapping2 = config["use_positional_encoding_mapping2"]
    number_of_positional_encoding_mapping2 = config["number_of_positional_encoding_mapping2"]
    number_of_layers_mapping2 = config["number_of_layers_mapping2"]
    number_of_channels_mapping2 = config["number_of_channels_mapping2"]

    # Atlas MLP's hyper parameters
    number_of_channels_atlas = config["number_of_channels_atlas"]
    number_of_layers_atlas = config["number_of_layers_atlas"]
    positional_encoding_num_atlas = config[
        "positional_encoding_num_atlas"]

    # bootstrapping configuration:
    alpha_bootstrapping_factor = config["alpha_bootstrapping_factor"]
    stop_bootstrapping_iteration = config["stop_bootstrapping_iteration"]

    # coefficients for the different loss terms
    rgb_coeff = config["rgb_coeff"] # coefficient for rgb loss term:
    alpha_flow_factor = config["alpha_flow_factor"]
    sparsity_coeff = config["sparsity_coeff"]
    # optical flow loss term coefficient (beta_f in the paper):
    optical_flow_coeff = config["optical_flow_coeff"]
    use_gradient_loss = config["use_gradient_loss"]
    gradient_loss_coeff = config["gradient_loss_coeff"]
    rigidity_coeff = config["rigidity_coeff"] # coefficient for the rigidity loss term
    derivative_amount = config["derivative_amount"]    # For finite differences gradient computation:
    # for using global (in addition to the current local) rigidity loss:
    include_global_rigidity_loss = config["include_global_rigidity_loss"]
    # Finite differences parameters for the global rigidity terms:
    global_rigidity_derivative_amount_fg = config["global_rigidity_derivative_amount_fg"]
    global_rigidity_derivative_amount_bg = config["global_rigidity_derivative_amount_bg"]
    global_rigidity_coeff_fg = config["global_rigidity_coeff_fg"]
    global_rigidity_coeff_bg = config["global_rigidity_coeff_bg"]
    stop_global_rigidity = config["stop_global_rigidity"]

    use_optical_flow = True

    vid_name = data_folder.name
    vid_root = data_folder.parent
    
    results_folder = Path(
        f'./{results_folder_name}/{vid_name}/stage_1')

    results_folder.mkdir(parents=True, exist_ok=True)
    with open('%s/config.json' % results_folder, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    
    writer = SummaryWriter(log_dir=str(results_folder))
    optical_flows_mask, video_frames, optical_flows_reverse_mask, mask_frames, video_frames_dx, video_frames_dy, optical_flows_reverse, optical_flows = load_input_data(
        resy, resx, maximum_number_of_frames, data_folder, True,  True, vid_root, vid_name)
    number_of_frames=video_frames.shape[3]
    # save a video showing the masked part of the forward optical flow:s
    save_mask_flow(optical_flows_mask, video_frames, results_folder)

    model_F_mapping1 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping1,
        use_positional=use_positional_encoding_mapping1,
        positional_dim=number_of_positional_encoding_mapping1,
        num_layers=number_of_layers_mapping1,
        skip_layers=[]).to(device)

    model_F_mapping2 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping2,
        use_positional=use_positional_encoding_mapping2,
        positional_dim=number_of_positional_encoding_mapping2,
        num_layers=number_of_layers_mapping2,
        skip_layers=[]).to(device)

    model_F_atlas = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=number_of_channels_atlas,
        use_positional=True,
        positional_dim=positional_encoding_num_atlas,
        num_layers=number_of_layers_atlas,
        skip_layers=[4, 7]).to(device)

    model_alpha = IMLP(
        input_dim=3,
        output_dim=1,
        hidden_dim=number_of_channels_alpha,
        use_positional=True,
        positional_dim=positional_encoding_num_alpha,
        num_layers=number_of_layers_alpha,
        skip_layers=[]).to(device)

    start_iteration = 0

    optimizer_all = optim.Adam(
        [{'params': list(model_F_mapping1.parameters())},
         {'params': list(model_F_mapping2.parameters())},
         {'params': list(model_alpha.parameters())},
         {'params': list(model_F_atlas.parameters())}], lr=0.0001)

    larger_dim = np.maximum(resx, resy)
    if not load_checkpoint:
        if pretrain_mapping1:
            model_F_mapping1 = pre_train_mapping(model_F_mapping1, number_of_frames, uv_mapping_scale, resx=resx, resy=resy,
                                                 larger_dim=larger_dim,device=device, pretrain_iters=pretrain_iter_number)
        if pretrain_mapping2:
            model_F_mapping2 = pre_train_mapping(model_F_mapping2, number_of_frames, uv_mapping_scale, resx=resx, resy=resy,
                                                 larger_dim=larger_dim, device=device,pretrain_iters=pretrain_iter_number)
    else:
        init_file = torch.load(checkpoint_path)
        model_F_atlas.load_state_dict(init_file["F_atlas_state_dict"])
        model_F_mapping1.load_state_dict(init_file["model_F_mapping1_state_dict"])
        model_F_mapping2.load_state_dict(init_file["model_F_mapping2_state_dict"])
        model_alpha.load_state_dict(init_file["model_F_alpha_state_dict"])
        optimizer_all.load_state_dict(init_file["optimizer_all_state_dict"])
        start_iteration = init_file["iteration"]

    jif_all = get_tuples(number_of_frames, video_frames)

    # Start training!
    for i in tqdm(range(start_iteration, iters_num)):

        if i > stop_bootstrapping_iteration:
            alpha_bootstrapping_factor = 0
        if i > stop_global_rigidity:
            global_rigidity_coeff_fg = 0
            global_rigidity_coeff_bg = 0
        # print(i)

        # randomly choose indices for the current batch
        inds_foreground = torch.randint(jif_all.shape[1],
                                        (np.int64(samples * 1.0), 1))

        jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)

        rgb_current = video_frames[jif_current[1, :], jif_current[0, :], :,
                      jif_current[2, :]].squeeze(1).to(device)

        # the correct alpha according to the precomputed maskrcnn
        alpha_maskrcnn = mask_frames[jif_current[1, :], jif_current[0, :],
                                     jif_current[2, :]].squeeze(1).to(device).unsqueeze(-1)

        # normalize coordinates to be in [-1,1]
        xyt_current = torch.cat(
            (jif_current[0, :] / (larger_dim / 2) - 1, jif_current[1, :] / (larger_dim / 2) - 1,
             jif_current[2, :] / (number_of_frames / 2.0) - 1),
            dim=1).to(device)  # size (batch, 3)

        # get the atlas UV coordinates from the two mapping networks;
        uv_foreground1 = model_F_mapping1(xyt_current)
        uv_foreground2 = model_F_mapping2(xyt_current)

        # map tanh output of the alpha network to the range (0,1) :
        alpha = 0.5 * (model_alpha(xyt_current) + 1.0)
        # prevent a situation of alpha=0, or alpha=1 (for the BCE loss that uses log(alpha),log(1-alpha) below)
        alpha = alpha * 0.99
        alpha = alpha + 0.001

        # Sample atlas values. Foreground colors are sampled from [0,1]x[0,1] and background colors are sampled from [-1,0]x[-1,0]
        # Note that the original [u,v] coorinates are in [-1,1]x[-1,1] for both networks
        rgb_output1 = (model_F_atlas(uv_foreground1 * 0.5 + 0.5) + 1.0) * 0.5
        rgb_output2 = (model_F_atlas(
            uv_foreground2 * 0.5 - 0.5) + 1.0) * 0.5
        # Reconstruct final colors from the two layers (using alpha)
        rgb_output_foreground = rgb_output1 * alpha + rgb_output2 * (1.0 - alpha)

        if use_gradient_loss:
            gradient_loss = get_gradient_loss(video_frames_dx, video_frames_dy, jif_current,
                                               model_F_mapping1, model_F_mapping2, model_F_atlas,
                                               rgb_output_foreground,device,resx,number_of_frames,model_alpha)
        else:
            gradient_loss = 0.0

        rgb_output_foreground_not = rgb_output1 * (1.0 - alpha)

        rgb_loss = (torch.norm(rgb_output_foreground - rgb_current, dim=1) ** 2).mean()

        rgb_loss_sparsity = (torch.norm(rgb_output_foreground_not, dim=1) ** 2).mean()

        rigidity_loss1 = get_rigidity_loss(
            jif_current,
            derivative_amount,
            larger_dim,
            number_of_frames,
            model_F_mapping1,
            uv_foreground1,device,
            uv_mapping_scale=uv_mapping_scale)
        rigidity_loss2 = get_rigidity_loss(
            jif_current,
            derivative_amount,
            larger_dim,
            number_of_frames,
            model_F_mapping2,
            uv_foreground2,device,
            uv_mapping_scale=uv_mapping_scale)

        if include_global_rigidity_loss and i <= stop_global_rigidity:
            global_rigidity_loss1 = get_rigidity_loss(
                jif_current,
                global_rigidity_derivative_amount_fg,
                larger_dim,
                number_of_frames,
                model_F_mapping1,
                uv_foreground1,device,
                uv_mapping_scale=uv_mapping_scale)
            global_rigidity_loss2 = get_rigidity_loss(
                jif_current,
                global_rigidity_derivative_amount_bg,
                larger_dim,
                number_of_frames,
                model_F_mapping2,
                uv_foreground2,device,
                uv_mapping_scale=uv_mapping_scale)

        flow_loss1 = get_optical_flow_loss(
            jif_current, uv_foreground1, optical_flows_reverse, optical_flows_reverse_mask, larger_dim,
            number_of_frames, model_F_mapping1, optical_flows, optical_flows_mask, uv_mapping_scale,device, use_alpha=True,
            alpha=alpha)

        flow_loss2 = get_optical_flow_loss(
            jif_current, uv_foreground2, optical_flows_reverse, optical_flows_reverse_mask, larger_dim,
            number_of_frames, model_F_mapping2, optical_flows, optical_flows_mask, uv_mapping_scale,device, use_alpha=True,
            alpha=1 - alpha)

        flow_alpha_loss = get_optical_flow_alpha_loss(model_alpha,
                                                      jif_current, alpha, optical_flows_reverse,
                                                      optical_flows_reverse_mask, larger_dim,
                                                      number_of_frames, optical_flows,
                                                      optical_flows_mask, device)

        alpha_bootstrapping_loss = torch.mean(
            -alpha_maskrcnn * torch.log(alpha) - (1 - alpha_maskrcnn) * torch.log(1 - alpha))

        if include_global_rigidity_loss and i <= stop_global_rigidity:
            loss = rigidity_coeff * (
                        rigidity_loss1 + rigidity_loss2) + global_rigidity_coeff_fg * global_rigidity_loss1 + global_rigidity_coeff_bg * global_rigidity_loss2 + \
                   rgb_loss * rgb_coeff + optical_flow_coeff * (
                               flow_loss1 + flow_loss2) + alpha_bootstrapping_loss * alpha_bootstrapping_factor + flow_alpha_loss * alpha_flow_factor + rgb_loss_sparsity * sparsity_coeff + gradient_loss * gradient_loss_coeff
        else:
            loss = rigidity_coeff * (rigidity_loss1 + rigidity_loss2) + rgb_loss * rgb_coeff + optical_flow_coeff * (
                        flow_loss1 + flow_loss2) + alpha_bootstrapping_loss * alpha_bootstrapping_factor + flow_alpha_loss * alpha_flow_factor + rgb_loss_sparsity * sparsity_coeff + gradient_loss * gradient_loss_coeff

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()

        # render and evaluate videos every N iterations
        if i % evaluate_every == 0 and i > start_iteration:
            evaluate_model(model_F_atlas, resx, resy, number_of_frames, model_F_mapping1,
                                       model_F_mapping2, model_alpha,
                                       video_frames, results_folder, i, mask_frames, optimizer_all,
                                       writer, vid_name, derivative_amount, uv_mapping_scale,
                                       optical_flows,
                                       optical_flows_mask,device)

            rgb_img = video_frames[:, :, :, 0].numpy()
            # writer.add_image('Input/rgb_0', rgb_img, i, dataformats='HWC')
            model_F_atlas.train()
            model_F_mapping1.train()
            model_F_mapping2.train()
            model_alpha.train()

if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config_flow_100.json")
    parser.add_argument('--vid_name', type=str, default="Around_the_world_in_1896_001")
    parser.add_argument('--root', type=str, default="data/test/")
    parser.add_argument('--down', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--class_name', type=str, default="portrait")
    args = parser.parse_args()
    
    # select_gpu = "1" # default use 0
    select_gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
    
    config_path = "src/config/%s" % args.config
    vid_path = os.path.join(args.root, args.vid_name)

    args.vid_path = vid_path
    
    # get flow using current video
    cmd = "python src/preprocess_optical_flow.py --vid-path %s --gpu %s " % (vid_path, select_gpu)
    print(cmd)
    subprocess.call(cmd, shell=True)
    
    # get mask using current video
    if args.class_name == "portrait":
        cmd = "python src/preprocess_mask_portrait.py --vid-path %s --gpu %s " % (vid_path, select_gpu)
        print(cmd)
        subprocess.call(cmd, shell=True)
    else:
        cmd = "python src/preprocess_mask_rcnn.py --vid-path %s --class_name %s --gpu %s " % (vid_path, args.class_name, select_gpu)
        print(cmd)
        subprocess.call(cmd, shell=True)
    
    with open(config_path) as f:
        main(json.load(f), args)