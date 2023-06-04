import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import List, Tuple

from ..modeling import Sam
from .amg import calculate_stability_score

class ImageEncoderOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the image encoder of Sam, with some functions modified to enable
    model tracing. Also supports extra options controlling what information. See
    the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        use_preprocess: bool,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.use_preprocess = use_preprocess
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float)
        self.image_encoder = model.image_encoder

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        if self.use_preprocess:
            input_image = self.preprocess(input_image)
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # permute channels
        x = torch.permute(x, (2, 0, 1))

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        # expand channels
        x = torch.unsqueeze(x, 0)
        return x