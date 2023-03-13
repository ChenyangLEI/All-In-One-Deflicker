import sys
sys.path.append('./PWC-Net/PyTorch/')
import os
import cv2
import torch
import numpy as np
from math import ceil
import time
from torch.autograd import Variable
# from scipy.ndimage import imread
from scipy.misc import imsave, imread

import models
import os
from glob import glob
import flowlib as fl
import torch.nn as nn
from PIL import Image as PILImage
from networks.resample2d_package.modules.resample2d import Resample2d


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        

    output = nn.functional.grid_sample(x, vgrid)
    # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    # mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1
    
    # return output*mask
    return output


def img2tensor(img):

    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t

def tensor2img(img_t):

    # img = img_t[0].detach().to("cpu").numpy()
    img = img_t[0].numpy()
    
    img = np.transpose(img, (1, 2, 0))

    return img

def compute_flow_magnitude(flow):

    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag

def compute_flow_gradients(flow):

    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))
    
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

def detect_occlusion(fw_flow, bw_flow):
    # inputs: flow_forward, flow_backward
    # return: occlusion mask 
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1
    
    # with torch.no_grad():

    ## convert to tensor
    fw_flow_t = img2tensor(fw_flow).cuda()
    bw_flow_t = img2tensor(bw_flow).cuda()

    ## warp fw-flow to img2
    flow_warping = Resample2d().cuda()
    fw_flow_w = flow_warping(Variable(fw_flow_t), Variable(bw_flow_t))

    fw_flow_w = fw_flow_w.detach().cpu()

    ## convert to numpy array
    fw_flow_w = tensor2img(fw_flow_w.data)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    
    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.ones((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 0

    return occlusion


def get_flow(net, im1_fn, im2_fn, flow_fn, type='RGB'):
    # inputs: img_s, img_t 
    # return: flow from img_s to img_t

    # start = time.time() 
    if type == 'RGB':
        im_all_forward = [imread(img) for img in [im1_fn, im2_fn]]        
    else:
        im_all_forward = [np.tile(imread(img, 'L')[:,:,np.newaxis],[1,1,3]) for img in [im1_fn, im2_fn]]
        print im_all_forward[0].shape,im_all_forward[1].shape
        im_all_forward = [im[:, :, :3] for im in im_all_forward]
    divisor = 64.
    H = im_all_forward[0].shape[0]
    W = im_all_forward[0].shape[1]

    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    for i in range(len(im_all_forward)):
       im_all_forward[i] = cv2.resize(im_all_forward[i], (W_, H_))

    for _i, _inputs in enumerate(im_all_forward):
    #    print(im_all_forward[_i].shape)
       im_all_forward[_i] = im_all_forward[_i][:, :, ::-1]
       im_all_forward[_i] = 1.0 * im_all_forward[_i]/255.0
    
       im_all_forward[_i] = np.transpose(im_all_forward[_i], (2, 0, 1))
       im_all_forward[_i] = torch.from_numpy(im_all_forward[_i])
       im_all_forward[_i] = im_all_forward[_i].expand(1, im_all_forward[_i].size()[0], im_all_forward[_i].size()[1], im_all_forward[_i].size()[2])  
       im_all_forward[_i] = im_all_forward[_i].float()
    
    im_all_forward = torch.autograd.Variable(torch.cat(im_all_forward,1).cuda(), volatile=True)
    # print(im_all_forward.shape)
    flo = net(im_all_forward)
    flo = flo[0] * 20.0
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size 
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
    u_ = cv2.resize(flo[:,:,0],(W,H))
    v_ = cv2.resize(flo[:,:,1],(W,H))
    u_ *= W/ float(W_)
    v_ *= H/ float(H_)
    flo = np.dstack((u_,v_))
    # print time.time()-start
    # print flow_fn
    # writeFlowFile(flow_fn, flo)

    # img2 = torch.from_numpy(np.array(PILImage.open(im2_fn))).unsqueeze(0).permute(0,3,1,2).float().cuda()

    # warp_img = warp(img2, torch.from_numpy(flo).unsqueeze(0).permute(0,3,1,2).cuda())
    # warp_img = warp_img.data.permute(0,2,3,1).squeeze(0).cpu().numpy()
    # imsave(flow_fn.replace('.flo','warp.png'), warp_img)

    # fl.save_flow_image(flo, flow_fn.replace('flo','png'))

    return flo

def get_flow_forward_backward(net, im1_fn, im2_fn, flow_fn, type='RGB'):
    # inputs: path of img_1, img_2
    # return: forward and backward flow between these two images

    flow_forward = get_flow(net, im1_fn, im2_fn, flow_fn, type=type)
    flow_backward = get_flow(net, im2_fn, im1_fn, flow_fn, type=type)

    return flow_forward, flow_backward


