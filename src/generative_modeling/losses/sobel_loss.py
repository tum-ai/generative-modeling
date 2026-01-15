import torch
import torch.nn.functional as F


def sobel_loss_2d(pred, target, loss_type="L2"):
    """reconstruction loss on sobel gradients"""
    B, C, H, W = pred.shape
    dtype = pred.dtype
    device = pred.device
    
    # kernel
    gx_kernel = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=dtype, device=device)
    gy_kernel = torch.tensor([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=dtype, device=device)
    sobel_kernel = torch.stack([gx_kernel, gy_kernel], dim=0).unsqueeze(1) # (2, 1, 3, 3)
    
    # pull channel dim into batch dim to have separate gradients per channel
    pred_reshaped = pred.reshape(B * C, 1, H, W)
    target_reshaped = target.reshape(B * C, 1, H, W)
    
    # compute gradients
    grad_pred = F.conv2d(pred_reshaped, sobel_kernel, padding=1)  # (B*C, 2, H, W)
    grad_target = F.conv2d(target_reshaped, sobel_kernel, padding=1)  # (B*C, 2, H, W)
    
    # reconstruction loss
    if loss_type == "L1":
        grad_loss = F.l1_loss(grad_pred, grad_target)
    elif loss_type == "L2":
        grad_loss = F.mse_loss(grad_pred, grad_target)
    else:
        raise ValueError(f"rec_loss must be 'L1' or 'L2', got {loss_type}")
    
    return grad_loss
