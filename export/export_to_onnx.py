import torch

from collections import OrderedDict
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT


encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )

param = torch.load("files/demo/pytorch_model/sam_vit_b_01ec64.pth")

d = OrderedDict()
for k in param:
    if "image_encoder" in k:
        d[k[14:]] = param[k]

encoder.load_state_dict(d)
encoder.eval()

x = torch.randn((1, 3, 1024, 1024))
torch.onnx.export(encoder,
                  x,
                  "files/demo/onnx/SAM_Intact_VITb.onnx",
                  opset_version=12,
                  input_names=["input"],
                  output_names=["output"])
