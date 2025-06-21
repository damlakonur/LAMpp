# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from einops import rearrange

__all__ = ['LPIPSLoss']


class LPIPSLoss(nn.Module):
    """
    Compute LPIPS loss between two images.
    """

    def __init__(self, device, prefetch: bool = False):
        super().__init__()
        self.device = torch.device(device)

        self._cache: dict[str, nn.Module] = {}
        if prefetch:
            self.prefetch_models()

    def _get_model(self, name: str) -> nn.Module:
        if name in self._cache:
            return self._cache[name]

        # --- construct LPIPS net (eager, NOT compiled) -----------------
        import warnings, lpips
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            net = lpips.LPIPS(net=name,
                              eval_mode=True, verbose=False).to(self.device)

        # move tiny shift / scale buffers to same device
        s_layer = net.scaling_layer
        s_layer.shift = s_layer.shift.to(self.device)
        s_layer.scale = s_layer.scale.to(self.device)

        net.requires_grad_(False)      # inference-only
        self._cache[name] = net
        return net

    def prefetch_models(self):
        _model_names = ['alex', 'vgg']
        for model_name in _model_names:
            self._get_model(model_name)

    def forward(self, x, y, is_training: bool = True, conf_sigma=None, only_sym_conf=False):
        """
        Assume images are 0-1 scaled and channel first.
        
        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
            is_training: whether to use VGG or AlexNet.
        
        Returns:
            Mean-reduced LPIPS loss across batch.
        """
        model_name = 'alex' if is_training else 'vgg'
        loss_fn = self._get_model(model_name)
        EPS = 1e-7
        if len(x.shape) == 5:
            N, M, C, H, W = x.shape
            x = x.reshape(N*M, C, H, W)
            y = y.reshape(N*M, C, H, W)
            image_loss = loss_fn(x, y, normalize=True)
            image_loss = image_loss.mean(dim=[1, 2, 3])
            batch_loss = image_loss.reshape(N, M).mean(dim=1)
            all_loss = batch_loss.mean()
        else:
            image_loss = loss_fn(x, y, normalize=True)
            if conf_sigma is not None:
                image_loss = image_loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            image_loss = image_loss.mean(dim=[1, 2, 3])
            all_loss = image_loss.mean()
        return all_loss
