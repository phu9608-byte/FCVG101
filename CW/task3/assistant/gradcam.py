"""Grad-CAM (Selvaraju et al., 2017) for the Task 2 FruitCNN.

For the predicted class c with logit y_c and the feature maps A^k of the last conv block:

    alpha_k = mean_{i,j} dy_c / dA^k_ij          (importance of channel k)
    L_c     = ReLU( sum_k alpha_k A^k )          (coarse heat map, 32x32 for a 128x128 input)

The map is upsampled to the photo size and overlaid, showing WHICH pixels drove the decision.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def last_conv_activation(model: nn.Module) -> nn.Module:
    """Module whose output is the last convolutional feature map (ReLU after the last Conv2d)."""
    modules = list(model.features) if hasattr(model, "features") else list(model.modules())
    last_conv = max(i for i, m in enumerate(modules) if isinstance(m, nn.Conv2d))
    if last_conv + 1 < len(modules) and isinstance(modules[last_conv + 1], nn.ReLU):
        return modules[last_conv + 1]
    return modules[last_conv]


def scores_and_cam(model: nn.Module, x: torch.Tensor, class_index: int | None = None):
    """Return (softmax scores [C], class index used, cam [h, w] in 0..1) for one input tensor [1, 3, H, W]."""
    store = {}
    layer = last_conv_activation(model)

    def forward_hook(_module, _inputs, output):
        store["A"] = output
        output.register_hook(lambda grad: store.__setitem__("dA", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            scores = torch.softmax(logits, dim=1)[0].detach()
            c = int(scores.argmax()) if class_index is None else int(class_index)
            logits[0, c].backward()
    finally:
        handle.remove()
    A, dA = store["A"].detach()[0], store["dA"].detach()[0]            # [K, h, w]
    weights = dA.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * A).sum(dim=0)).cpu().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return scores.cpu().numpy(), c, cam


def overlay(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Upsample ``cam`` to the image and blend a JET colour map over it (red = most important)."""
    h, w = bgr.shape[:2]
    heat = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)
