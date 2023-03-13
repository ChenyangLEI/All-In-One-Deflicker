import torch

# gather the corresponding forward and backward point
def gather_corresponding_flow_matches(jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames, is_forward, boundary):
    # deal with boundary value
    jif_foreground[1][jif_foreground[1] == boundary] = boundary-1 # I dont's know why?
    jif_foreground[0][jif_foreground[0] == resx] = resx-1
    
    batch_forward_mask = torch.where(
        optical_flows_mask[jif_foreground[1, :].squeeze(), jif_foreground[0, :].squeeze(),
        jif_foreground[2, :].squeeze(), :])
    forward_frames_amount = 2 ** batch_forward_mask[1]
    relevant_batch_indices = batch_forward_mask[0]
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices, 0]
    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                             jif_foreground_forward_relevant[2], batch_forward_mask[1]]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] + forward_frames_amount))
    else:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] - forward_frames_amount))

    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                               number_of_frames / 2) - 1)).T
    
    return jif_foreground_forward_should_match, xyt_foreground_forward_should_match, relevant_batch_indices

def gather_corresponding_points(jif_init, optical_flows_mask, optical_flows, larger_dim, number_of_frames,
                      optical_flows_reverse_mask, optical_flows_reverse, average_frames, boundary):
    # boundary is a boundary value
    
    samples = jif_init.shape[1]
    # init
    xyt_init = torch.cat(
                (jif_init[0, :] / (larger_dim / 2) - 1, jif_init[1, :] / (larger_dim / 2) - 1,
                 jif_init[2, :] / (number_of_frames / 2.0) - 1), dim=1) # size (batch, 3)
    mask_init = torch.ones(samples, 1)
    
    # begin
    jif_list = []
    xyt_list = []
    relevant_batch_indices_list = []
    mask_list = []
    shape_list = []

    # forward
    jif_current = jif_init
    for _ in range(average_frames):
        # first get correspondence point
        jif_foreground_forward_should_match, xyt_foreground_forward_should_match, relevant_batch_indices_forward = gather_corresponding_flow_matches(
            jif_current, optical_flows_mask, optical_flows, larger_dim, number_of_frames, True, boundary)

        # init mask
        current_mask = torch.zeros(jif_current.shape[1], 1)
        current_mask[relevant_batch_indices_forward] = 1

        # backward to get the full mask
        for prev_shape, prev_indice in zip(shape_list[::-1], relevant_batch_indices_list[::-1]):
            temp_mask = torch.zeros(prev_shape, 1)
            temp_mask[prev_indice] = current_mask
            current_mask = temp_mask

        shape_list.append(jif_current.shape[1])
        mask_list.append(current_mask)

        jif_current = torch.round(jif_foreground_forward_should_match).type(torch.LongTensor)[:, :, None]
        # print(jif_current.shape)

        jif_list.append(jif_current)
        xyt_list.append(xyt_foreground_forward_should_match)
        relevant_batch_indices_list.append(relevant_batch_indices_forward)
    # add the final one
    shape_list.append(jif_current.shape[1])

    # backward
    jif_list_back = []
    xyt_list_back = []
    relevant_batch_indices_list_back = []
    mask_list_back = []
    shape_list_back = []
    jif_current = jif_init
    for _ in range(average_frames):
        # first get correspondence point
        jif_foreground_forward_should_match, xyt_foreground_forward_should_match, relevant_batch_indices_forward = gather_corresponding_flow_matches(
            jif_current, optical_flows_reverse_mask, optical_flows_reverse, larger_dim, number_of_frames, False, boundary)

        # init mask
        current_mask = torch.zeros(jif_current.shape[1], 1)
        current_mask[relevant_batch_indices_forward] = 1

        # backward to get the full mask
        for prev_shape, prev_indice in zip(shape_list_back[::-1], relevant_batch_indices_list_back[::-1]):
            temp_mask = torch.zeros(prev_shape, 1)
            temp_mask[prev_indice] = current_mask
            current_mask = temp_mask

        shape_list_back.append(jif_current.shape[1])
        mask_list_back.append(current_mask)

        jif_current = torch.round(jif_foreground_forward_should_match).type(torch.LongTensor)[:, :, None]
        # print(jif_current.shape)

        jif_list_back.append(jif_current)
        xyt_list_back.append(xyt_foreground_forward_should_match)
        relevant_batch_indices_list_back.append(relevant_batch_indices_forward)
    # add the final one
    shape_list_back.append(jif_current.shape[1])

    # add the initial one to the list and then concat
    # shape
    shape_concat = shape_list + shape_list_back[1::]
    # jif
    jif_list.insert(0, jif_init)
    jif_concat = torch.cat(jif_list + jif_list_back, 1) 
    # xyt
    xyt_list.insert(0, xyt_init)
    xyt_concat = torch.cat(xyt_list + xyt_list_back, 0) 
    # mask
    mask_list.insert(0, mask_init)
    mask_concat = torch.stack(mask_list + mask_list_back, 0) 

    return shape_concat, jif_concat, xyt_concat, mask_concat


# calculating the gradient loss as defined by Eq.7 in the paper for only one mapping network
def get_gradient_loss_single(video_frames_dx, video_frames_dy, jif_current,
                      model_F_mapping1, model_F_atlas,
                      rgb_output_foreground, device,resx,number_of_frames):
    xplus1yt_foreground = torch.cat(
        ((jif_current[0, :] + 1) / (resx / 2) - 1, jif_current[1, :] / (resx / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    xyplus1t_foreground = torch.cat(
        ((jif_current[0, :]) / (resx / 2) - 1, (jif_current[1, :] + 1) / (resx / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    # precomputed discrete derivative with respect to x,y direction
    rgb_dx_gt = video_frames_dx[jif_current[1, :], jif_current[0, :], :,
                jif_current[2, :]].squeeze(1).to(device)
    rgb_dy_gt = video_frames_dy[jif_current[1, :], jif_current[0, :], :,
                jif_current[2, :]].squeeze(1).to(device)

    # uv coordinates for locations with offsets of 1 pixel
    uv_foreground1_xyplus1t = model_F_mapping1(xyplus1t_foreground)
    uv_foreground1_xplus1yt = model_F_mapping1(xplus1yt_foreground)

    # The RGB values (from the 2 layers) for locations with offsets of 1 pixel
    rgb_output1_xyplus1t = (model_F_atlas(uv_foreground1_xyplus1t * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output1_xplus1yt = (model_F_atlas(uv_foreground1_xplus1yt * 0.5 + 0.5) + 1.0) * 0.5

    # Reconstructed RGB values:
    rgb_output_foreground_xyplus1t = rgb_output1_xyplus1t
    rgb_output_foreground_xplus1yt = rgb_output1_xplus1yt

    # Use reconstructed RGB values for computing derivatives:
    rgb_dx_output = rgb_output_foreground_xplus1yt - rgb_output_foreground
    rgb_dy_output = rgb_output_foreground_xyplus1t - rgb_output_foreground
    gradient_loss = torch.mean(
        (rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2 + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2)
    return gradient_loss

# calculating the gradient loss as defined by Eq.7 in the paper
def get_gradient_loss(video_frames_dx, video_frames_dy, jif_current,
                      model_F_mapping1, model_F_mapping2, model_F_atlas,
                      rgb_output_foreground, device,resx,number_of_frames,model_alpha):
    xplus1yt_foreground = torch.cat(
        ((jif_current[0, :] + 1) / (resx / 2) - 1, jif_current[1, :] / (resx / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    xyplus1t_foreground = torch.cat(
        ((jif_current[0, :]) / (resx / 2) - 1, (jif_current[1, :] + 1) / (resx / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    alphaxplus1 = 0.5 * (model_alpha(xplus1yt_foreground) + 1.0)
    alphaxplus1 = alphaxplus1 * 0.99
    alphaxplus1 = alphaxplus1 + 0.001

    alphayplus1 = 0.5 * (model_alpha(xyplus1t_foreground) + 1.0)
    alphayplus1 = alphayplus1 * 0.99
    alphayplus1 = alphayplus1 + 0.001


    # precomputed discrete derivative with respect to x,y direction
    rgb_dx_gt = video_frames_dx[jif_current[1, :], jif_current[0, :], :,
                jif_current[2, :]].squeeze(1).to(device)
    rgb_dy_gt = video_frames_dy[jif_current[1, :], jif_current[0, :], :,
                jif_current[2, :]].squeeze(1).to(device)

    # uv coordinates for locations with offsets of 1 pixel
    uv_foreground2_xyplus1t = model_F_mapping2(xyplus1t_foreground)
    uv_foreground2_xplus1yt = model_F_mapping2(xplus1yt_foreground)
    uv_foreground1_xyplus1t = model_F_mapping1(xyplus1t_foreground)
    uv_foreground1_xplus1yt = model_F_mapping1(xplus1yt_foreground)

    # The RGB values (from the 2 layers) for locations with offsets of 1 pixel
    rgb_output1_xyplus1t = (model_F_atlas(uv_foreground1_xyplus1t * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output1_xplus1yt = (model_F_atlas(uv_foreground1_xplus1yt * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output2_xyplus1t = (model_F_atlas(uv_foreground2_xyplus1t * 0.5 - 0.5) + 1.0) * 0.5
    rgb_output2_xplus1yt = (model_F_atlas(uv_foreground2_xplus1yt * 0.5 - 0.5) + 1.0) * 0.5

    # Reconstructed RGB values:
    rgb_output_foreground_xyplus1t = rgb_output1_xyplus1t * alphayplus1 + rgb_output2_xyplus1t * (
            1.0 - alphayplus1)
    rgb_output_foreground_xplus1yt = rgb_output1_xplus1yt * alphaxplus1 + rgb_output2_xplus1yt * (
            1.0 - alphaxplus1)

    # Use reconstructed RGB values for computing derivatives:
    rgb_dx_output = rgb_output_foreground_xplus1yt - rgb_output_foreground
    rgb_dy_output = rgb_output_foreground_xyplus1t - rgb_output_foreground
    gradient_loss = torch.mean(
        (rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2 + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2)
    return gradient_loss

# get rigidity loss as defined in Eq. 9 in the paper
def get_rigidity_loss(jif_foreground, derivative_amount, resx, number_of_frames, model_F_mapping, uv_foreground, device,
                      uv_mapping_scale=1.0, return_all=False):
    # concatenating (x,y-derivative_amount,t) and (x-derivative_amount,y,t) to get xyt_p:
    is_patch = torch.cat((jif_foreground[1, :] - derivative_amount, jif_foreground[1, :])) / (resx / 2) - 1
    js_patch = torch.cat((jif_foreground[0, :], jif_foreground[0, :] - derivative_amount)) / (resx / 2) - 1
    fs_patch = torch.cat((jif_foreground[2, :], jif_foreground[2, :])) / (number_of_frames / 2.0) - 1
    xyt_p = torch.cat((js_patch, is_patch, fs_patch), dim=1).to(device)

    uv_p = model_F_mapping(xyt_p)
    u_p = uv_p[:, 0].view(2, -1)  # u_p[0,:]= u(x,y-derivative_amount,t).  u_p[1,:]= u(x-derivative_amount,y,t)
    v_p = uv_p[:, 1].view(2, -1)  # v_p[0,:]= u(x,y-derivative_amount,t).  v_p[1,:]= v(x-derivative_amount,y,t)

    u_p_d_ = uv_foreground[:, 0].unsqueeze(
        0) - u_p  # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    v_p_d_ = uv_foreground[:, 1].unsqueeze(
        0) - v_p  # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).

    # to match units: 1 in uv coordinates is resx/2 in image space.
    du_dx = u_p_d_[1, :] * resx / 2
    du_dy = u_p_d_[0, :] * resx / 2
    dv_dy = v_p_d_[0, :] * resx / 2
    dv_dx = v_p_d_[1, :] * resx / 2

    jacobians = torch.cat((torch.cat((du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)), dim=2),
                           torch.cat((dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)),
                                     dim=2)),
                          dim=1)
    jacobians = jacobians / uv_mapping_scale
    jacobians = jacobians / derivative_amount

    # Apply a loss to constrain the Jacobian to be a rotation matrix as much as possible
    JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

    a = JtJ[:, 0, 0] + 0.001
    b = JtJ[:, 0, 1]
    c = JtJ[:, 1, 0]
    d = JtJ[:, 1, 1] + 0.001

    JTJinv = torch.zeros_like(jacobians).to(device)
    JTJinv[:, 0, 0] = d
    JTJinv[:, 0, 1] = -b
    JTJinv[:, 1, 0] = -c
    JTJinv[:, 1, 1] = a
    JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

    # See Equation (9) in the paper:
    rigidity_loss = (JtJ ** 2).sum(1).sum(1).sqrt() + (JTJinv ** 2).sum(1).sum(1).sqrt()

    if return_all:
        return rigidity_loss
    else:
        return rigidity_loss.mean()



# Compute optical flow loss (Eq. 11 in the paper) for all pixels without averaging. This is relevant for visualization of the loss.
def get_optical_flow_loss_all(jif_foreground, uv_foreground,
                              resx, number_of_frames, model_F_mapping,
                              optical_flows, optical_flows_mask, uv_mapping_scale, device,
                              alpha=1.0):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames)
    uv_foreground_forward_should_match = model_F_mapping(xyt_foreground_forward_should_match.to(device))

    errors = (uv_foreground_forward_should_match - uv_foreground).norm(dim=1)
    errors[relevant_batch_indices_forward == False] = 0
    errors = errors * (alpha.squeeze())

    return errors * resx / (2 * uv_mapping_scale)


# Compute optical flow loss (Eq. 11 in the paper)
def get_optical_flow_loss(jif_foreground, uv_foreground, optical_flows_reverse, optical_flows_reverse_mask, resx,
                          number_of_frames, model_F_mapping, optical_flows, optical_flows_mask, uv_mapping_scale,
                          device, use_alpha=False, alpha=1.0):
    # Forward flow:
    uv_foreground_forward_relevant, xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames, True, uv_foreground)
    uv_foreground_forward_should_match = model_F_mapping(xyt_foreground_forward_should_match.to(device))
    loss_flow_next = (uv_foreground_forward_should_match - uv_foreground_forward_relevant).norm(dim=1) * resx / (
                2 * uv_mapping_scale)

    # Backward flow:
    uv_foreground_backward_relevant, xyt_foreground_backward_should_match, relevant_batch_indices_backward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_reverse_mask, optical_flows_reverse, resx, number_of_frames, False, uv_foreground)
    uv_foreground_backward_should_match = model_F_mapping(xyt_foreground_backward_should_match.to(device))
    loss_flow_prev = (uv_foreground_backward_should_match - uv_foreground_backward_relevant).norm(dim=1) * resx / (
                2 * uv_mapping_scale)

    if use_alpha:
        flow_loss = (loss_flow_prev * alpha[relevant_batch_indices_backward].squeeze()).mean() * 0.5 + (
                    loss_flow_next * alpha[relevant_batch_indices_forward].squeeze()).mean() * 0.5
    else:
        flow_loss = (loss_flow_prev).mean() * 0.5 + (loss_flow_next).mean() * 0.5

    return flow_loss


# A helper function for get_optical_flow_loss to return matching points according to the optical flow
def get_corresponding_flow_matches(jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames,
                                   is_forward, uv_foreground, use_uv=True):
    batch_forward_mask = torch.where(
        optical_flows_mask[jif_foreground[1, :].squeeze(), jif_foreground[0, :].squeeze(),
        jif_foreground[2, :].squeeze(), :])
    forward_frames_amount = 2 ** batch_forward_mask[1]
    relevant_batch_indices = batch_forward_mask[0]
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices, 0]
    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                             jif_foreground_forward_relevant[2], batch_forward_mask[1]]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] + forward_frames_amount))
    else:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] - forward_frames_amount))

    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                               number_of_frames / 2) - 1)).T
    if use_uv:
        uv_foreground_forward_relevant = uv_foreground[batch_forward_mask[0]]
        return uv_foreground_forward_relevant, xyt_foreground_forward_should_match, relevant_batch_indices
    else:
        return xyt_foreground_forward_should_match, relevant_batch_indices


# A helper function for get_optical_flow_loss_all to return matching points according to the optical flow
def get_corresponding_flow_matches_all(jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames,
                                        use_uv=True):
    jif_foreground_forward_relevant = jif_foreground

    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                             jif_foreground_forward_relevant[2], 0].squeeze()
    forward_flows_for_loss_mask = optical_flows_mask[
        jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0],
        jif_foreground_forward_relevant[2], 0].squeeze()

    jif_foreground_forward_should_match = torch.stack(
        (jif_foreground_forward_relevant[0].squeeze() + forward_flows_for_loss[:, 0],
         jif_foreground_forward_relevant[1].squeeze() + forward_flows_for_loss[:, 1],
         jif_foreground_forward_relevant[2].squeeze() + 1))

    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                               number_of_frames / 2) - 1)).T
    if use_uv:
        return xyt_foreground_forward_should_match, forward_flows_for_loss_mask > 0
    else:
        return 0

# Compute alpha optical flow loss (Eq. 12 in the paper)
def get_optical_flow_alpha_loss(model_alpha,
                                jif_foreground, alpha, optical_flows_reverse, optical_flows_reverse_mask, resx,
                                number_of_frames, optical_flows, optical_flows_mask, device):
    # Forward flow
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames, True, 0, use_uv=False)
    alpha_foreground_forward_should_match = model_alpha(xyt_foreground_forward_should_match.to(device))
    alpha_foreground_forward_should_match = 0.5 * (alpha_foreground_forward_should_match + 1.0)
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match + 0.001
    loss_flow_alpha_next = (alpha[relevant_batch_indices_forward] - alpha_foreground_forward_should_match).abs().mean()

    # Backward loss
    xyt_foreground_backward_should_match, relevant_batch_indices_backward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_reverse_mask, optical_flows_reverse, resx, number_of_frames, False, 0,
        use_uv=False)
    alpha_foreground_backward_should_match = model_alpha(xyt_foreground_backward_should_match.to(device))
    alpha_foreground_backward_should_match = 0.5 * (alpha_foreground_backward_should_match + 1.0)
    alpha_foreground_backward_should_match = alpha_foreground_backward_should_match * 0.99
    alpha_foreground_backward_should_match = alpha_foreground_backward_should_match + 0.001
    loss_flow_alpha_prev = (
                alpha_foreground_backward_should_match - alpha[relevant_batch_indices_backward]).abs().mean()

    return (loss_flow_alpha_next + loss_flow_alpha_prev) * 0.5


# Compute alpha optical flow loss (Eq. 12 in the paper) for all the pixels for visualization.
def get_optical_flow_alpha_loss_all(model_alpha,
                                    jif_foreground, alpha, resx,
                                    number_of_frames, optical_flows, optical_flows_mask, device):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames)
    alpha_foreground_forward_should_match = model_alpha(xyt_foreground_forward_should_match.to(device))
    alpha_foreground_forward_should_match = 0.5 * (alpha_foreground_forward_should_match + 1.0)
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match + 0.001

    loss_flow_alpha_next = (alpha - alpha_foreground_forward_should_match).abs()
    loss_flow_alpha_next[relevant_batch_indices_forward == False] = 0

    return loss_flow_alpha_next
