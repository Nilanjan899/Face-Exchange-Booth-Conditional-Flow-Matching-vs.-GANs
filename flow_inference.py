# flow_inference.py
#
# Conditional Rectified Flow / Flow Matching inference library
# Matching:
#   conditional-rectified-flow-matching-paired-face (5).ipynb
#
# API:
#
#   from flow_inference import load_flow_pipeline
#
#   flow = load_flow_pipeline(
#       checkpoint=r"D:\models\flow_checkpoint.zip",
#       bisenet_checkpoint=r"D:\models\79999_iter.pth"
#   )
#
#   result = flow(
#       source=r"D:\images\source.jpg",
#       target=r"D:\images\target.jpg"
#   )
#
#   result["image"].show()


import os
import math
import zipfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from facenet_pytorch import InceptionResnetV1


# ============================================================
# Constants
# ============================================================

IMG_SIZE = 512
CROP_SIZE = 256

DEFAULT_SAMPLE_STEPS = 4
DEFAULT_SOLVER = "heun"

FACE_CLASS_IDS = [
    1, 2, 3, 4, 5, 6,
    10, 11, 12, 13
]


# ============================================================
# Utility
# ============================================================

def _select_device(device=None):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _groups(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def _load_rgb_tensor(path, size=IMG_SIZE):
    """
    Exactly matches the notebook's load_resized_tensor():

        RGB
        resize -> 512x512
        [0,255] -> [0,1]
        HWC -> CHW
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(
            (size, size),
            Image.Resampling.BILINEAR
        )

        arr = np.asarray(img, dtype=np.uint8)

    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
    )


# ============================================================
# BiSeNet
# Exact architecture used by the training notebook
# ============================================================

class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            3,
            stride,
            1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_chan)

        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            3,
            1,
            1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_chan)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None

        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    1,
                    stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):

        residual = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        residual = self.bn2(
            self.conv2(residual)
        )

        shortcut = (
            self.downsample(x)
            if self.downsample is not None
            else x
        )

        return self.relu(shortcut + residual)


def create_layer_basic(
    in_chan,
    out_chan,
    bnum,
    stride=1
):

    layers = [
        BasicBlock(
            in_chan,
            out_chan,
            stride=stride
        )
    ]

    for _ in range(bnum - 1):
        layers.append(
            BasicBlock(
                out_chan,
                out_chan,
                1
            )
        )

    return nn.Sequential(*layers)


class Resnet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            7,
            2,
            3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(
            3,
            2,
            1
        )

        self.layer1 = create_layer_basic(
            64, 64, 2, 1
        )

        self.layer2 = create_layer_basic(
            64, 128, 2, 2
        )

        self.layer3 = create_layer_basic(
            128, 256, 2, 2
        )

        self.layer4 = create_layer_basic(
            256, 512, 2, 2
        )

    def forward(self, x):

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        return feat8, feat16, feat32


class ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            ks,
            stride,
            padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return F.relu(
            self.bn(
                self.conv(x)
            )
        )


class BiSeNetOutput(nn.Module):

    def __init__(
        self,
        in_chan,
        mid_chan,
        n_classes
    ):

        super().__init__()

        self.conv = ConvBNReLU(
            in_chan,
            mid_chan
        )

        self.conv_out = nn.Conv2d(
            mid_chan,
            n_classes,
            1,
            bias=False
        )

    def forward(self, x):
        return self.conv_out(
            self.conv(x)
        )


class AttentionRefinementModule(nn.Module):

    def __init__(
        self,
        in_chan,
        out_chan
    ):

        super().__init__()

        self.conv = ConvBNReLU(
            in_chan,
            out_chan
        )

        self.conv_atten = nn.Conv2d(
            out_chan,
            out_chan,
            1,
            bias=False
        )

        self.bn_atten = nn.BatchNorm2d(
            out_chan
        )

    def forward(self, x):

        feat = self.conv(x)

        atten = F.avg_pool2d(
            feat,
            feat.size()[2:]
        )

        atten = torch.sigmoid(
            self.bn_atten(
                self.conv_atten(atten)
            )
        )

        return feat * atten


class ContextPath(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = Resnet18()

        self.arm16 = AttentionRefinementModule(
            256,
            128
        )

        self.arm32 = AttentionRefinementModule(
            512,
            128
        )

        self.conv_head32 = ConvBNReLU(
            128,
            128
        )

        self.conv_head16 = ConvBNReLU(
            128,
            128
        )

        self.conv_avg = ConvBNReLU(
            512,
            128,
            ks=1,
            stride=1,
            padding=0
        )

    def forward(self, x):

        feat8, feat16, feat32 = self.resnet(x)

        H8, W8 = feat8.shape[2:]
        H16, W16 = feat16.shape[2:]
        H32, W32 = feat32.shape[2:]

        avg = self.conv_avg(
            F.avg_pool2d(
                feat32,
                feat32.size()[2:]
            )
        )

        avg_up = F.interpolate(
            avg,
            (H32, W32),
            mode="nearest"
        )

        feat32_up = self.conv_head32(
            F.interpolate(
                self.arm32(feat32) + avg_up,
                (H16, W16),
                mode="nearest"
            )
        )

        feat16_up = self.conv_head16(
            F.interpolate(
                self.arm16(feat16) + feat32_up,
                (H8, W8),
                mode="nearest"
            )
        )

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):

    def __init__(
        self,
        in_chan,
        out_chan
    ):

        super().__init__()

        self.convblk = ConvBNReLU(
            in_chan,
            out_chan,
            ks=1,
            stride=1,
            padding=0
        )

        self.conv1 = nn.Conv2d(
            out_chan,
            out_chan // 4,
            1,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            out_chan // 4,
            out_chan,
            1,
            bias=False
        )

    def forward(self, fsp, fcp):

        feat = self.convblk(
            torch.cat([fsp, fcp], dim=1)
        )

        atten = F.avg_pool2d(
            feat,
            feat.size()[2:]
        )

        atten = torch.sigmoid(
            self.conv2(
                F.relu(
                    self.conv1(atten)
                )
            )
        )

        return feat * atten + feat


class BiSeNet(nn.Module):

    def __init__(self, n_classes=19):

        super().__init__()

        self.cp = ContextPath()

        self.ffm = FeatureFusionModule(
            256,
            256
        )

        self.conv_out = BiSeNetOutput(
            256,
            256,
            n_classes
        )

        self.conv_out16 = BiSeNetOutput(
            128,
            64,
            n_classes
        )

        self.conv_out32 = BiSeNetOutput(
            128,
            64,
            n_classes
        )

    def forward(self, x):

        H, W = x.shape[2:]

        feat_res8, feat_cp8, feat_cp16 = self.cp(x)

        out = self.conv_out(
            self.ffm(
                feat_res8,
                feat_cp8
            )
        )

        out16 = self.conv_out16(feat_cp8)
        out32 = self.conv_out32(feat_cp16)

        return (
            F.interpolate(
                out,
                (H, W),
                mode="bilinear",
                align_corners=True
            ),

            F.interpolate(
                out16,
                (H, W),
                mode="bilinear",
                align_corners=True
            ),

            F.interpolate(
                out32,
                (H, W),
                mode="bilinear",
                align_corners=True
            ),
        )


class BiSeNetMaskExtractor(nn.Module):

    def __init__(
        self,
        bisenet,
        dilate_px=6
    ):

        super().__init__()

        self.bisenet = bisenet.eval()

        for p in self.bisenet.parameters():
            p.requires_grad_(False)

        self.dilate_px = dilate_px

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    @torch.inference_mode()
    def forward(self, img):

        B, C, H, W = img.shape

        x = F.interpolate(
            img,
            (512, 512),
            mode="bilinear",
            align_corners=False
        )

        logits, _, _ = self.bisenet(
            (x - self.mean) / self.std
        )

        parsing = logits.argmax(
            1,
            keepdim=True
        )

        mask = torch.zeros_like(
            parsing,
            dtype=torch.float32
        )

        for cls_id in FACE_CLASS_IDS:
            mask += (
                parsing == cls_id
            ).float()

        mask = mask.clamp(0, 1)

        mask = F.interpolate(
            mask,
            (H, W),
            mode="bilinear",
            align_corners=False
        )

        if self.dilate_px:

            k = self.dilate_px * 2 + 1

            mask = kornia.morphology.dilation(
                mask,
                torch.ones(
                    (k, k),
                    device=img.device
                )
            )

        return mask.clamp(0, 1)


def mask_to_bbox(
    mask,
    margin_ratio=0.35
):

    B, _, H, W = mask.shape

    boxes = []

    bin_mask = mask > 0.5

    for b in range(B):

        ys, xs = torch.where(
            bin_mask[b, 0]
        )

        if len(ys) == 0:

            size = min(H, W)

            boxes.append([
                H // 2 - size // 2,
                W // 2 - size // 2,
                size
            ])

            continue

        y0, y1 = (
            ys.min().item(),
            ys.max().item()
        )

        x0, x1 = (
            xs.min().item(),
            xs.max().item()
        )

        h = max(1, y1 - y0)
        w = max(1, x1 - x0)

        size = max(
            8,
            int(
                max(h, w) *
                (1 + margin_ratio)
            )
        )

        cy, cx = (
            (y0 + y1) // 2,
            (x0 + x1) // 2
        )

        top = max(
            0,
            min(
                H - size,
                cy - size // 2
            )
        )

        left = max(
            0,
            min(
                W - size,
                cx - size // 2
            )
        )

        size = min(
            size,
            H - top,
            W - left
        )

        boxes.append([
            int(top),
            int(left),
            int(size)
        ])

    return boxes


# ============================================================
# Flow Matching model
# Exact model from the training notebook
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):

        half = self.dim // 2

        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half,
                device=t.device
            ).float()
            / max(half - 1, 1)
        )

        x = (
            t.float()[:, None]
            * freq[None, :]
        )

        emb = torch.cat(
            [
                torch.sin(x),
                torch.cos(x)
            ],
            dim=1
        )

        if self.dim % 2:
            emb = F.pad(
                emb,
                (0, 1)
            )

        return emb


class ConvGNAct(nn.Module):

    def __init__(
        self,
        in_c,
        out_c,
        stride=1
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_c,
            out_c,
            3,
            stride,
            1,
            bias=False
        )

        self.gn = nn.GroupNorm(
            _groups(out_c),
            out_c
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(self, x):

        return self.act(
            self.gn(
                self.conv(x)
            )
        )


class ConditionalResBlock(nn.Module):

    def __init__(
        self,
        channels,
        cond_dim
    ):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            _groups(channels),
            channels
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(
            _groups(channels),
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1,
            bias=False
        )

        self.cond = nn.Linear(
            cond_dim,
            channels * 2
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(self, x, cond):

        h = self.conv1(
            self.norm1(x)
        )

        scale, bias = self.cond(
            cond
        ).chunk(2, dim=1)

        h = (
            self.norm2(h)
            *
            (
                1.0
                +
                0.1
                *
                torch.tanh(
                    scale[..., None, None]
                )
            )
            +
            bias[..., None, None]
        )

        h = self.act(h)

        h = self.conv2(h)

        return x + h


class IdentityCrossAttention(nn.Module):

    def __init__(
        self,
        channels,
        identity_dim=512,
        heads=4,
        tokens=4
    ):

        super().__init__()

        self.tokens = tokens

        self.id_to_tokens = nn.Sequential(
            nn.Linear(
                identity_dim,
                channels * tokens
            ),
            nn.SiLU(),
            nn.Linear(
                channels * tokens,
                channels * tokens
            )
        )

        self.norm = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(
            channels,
            heads,
            batch_first=True,
            dropout=0.0
        )

        self.proj = nn.Linear(
            channels,
            channels
        )

    def forward(self, x, identity):

        B, C, H, W = x.shape

        q = (
            x.flatten(2)
            .transpose(1, 2)
        )

        qn = self.norm(q)

        tokens = self.id_to_tokens(
            identity
        ).view(
            B,
            self.tokens,
            C
        )

        out, _ = self.attn(
            qn,
            tokens,
            tokens,
            need_weights=False
        )

        out = (
            self.proj(out)
            .transpose(1, 2)
            .view(B, C, H, W)
        )

        return x + 0.5 * out


class FlowMatchingUNet(nn.Module):

    def __init__(
        self,
        base=32,
        cond_dim=128
    ):

        super().__init__()

        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(
                cond_dim,
                cond_dim
            ),
            nn.SiLU(),
            nn.Linear(
                cond_dim,
                cond_dim
            )
        )

        self.id_proj = nn.Sequential(
            nn.Linear(
                512,
                cond_dim
            ),
            nn.SiLU(),
            nn.Linear(
                cond_dim,
                cond_dim
            )
        )

        self.in_conv = ConvGNAct(
            6,
            base
        )

        self.e1 = nn.ModuleList([
            ConditionalResBlock(
                base,
                cond_dim
            ),
            ConditionalResBlock(
                base,
                cond_dim
            )
        ])

        self.down1 = ConvGNAct(
            base,
            base * 2,
            stride=2
        )

        self.e2 = nn.ModuleList([
            ConditionalResBlock(
                base * 2,
                cond_dim
            ),
            ConditionalResBlock(
                base * 2,
                cond_dim
            )
        ])

        self.down2 = ConvGNAct(
            base * 2,
            base * 3,
            stride=2
        )

        self.e3 = nn.ModuleList([
            ConditionalResBlock(
                base * 3,
                cond_dim
            ),
            ConditionalResBlock(
                base * 3,
                cond_dim
            )
        ])

        self.down3 = ConvGNAct(
            base * 3,
            base * 4,
            stride=2
        )

        self.mid = nn.ModuleList([
            ConditionalResBlock(
                base * 4,
                cond_dim
            ),
            ConditionalResBlock(
                base * 4,
                cond_dim
            )
        ])

        self.cross_attn = IdentityCrossAttention(
            base * 4,
            identity_dim=512
        )

        self.up3 = ConvGNAct(
            base * 4,
            base * 3
        )

        self.d3 = nn.ModuleList([
            ConditionalResBlock(
                base * 3,
                cond_dim
            ),
            ConditionalResBlock(
                base * 3,
                cond_dim
            )
        ])

        self.up2 = ConvGNAct(
            base * 3,
            base * 2
        )

        self.d2 = nn.ModuleList([
            ConditionalResBlock(
                base * 2,
                cond_dim
            ),
            ConditionalResBlock(
                base * 2,
                cond_dim
            )
        ])

        self.up1 = ConvGNAct(
            base * 2,
            base
        )

        self.d1 = nn.ModuleList([
            ConditionalResBlock(
                base,
                cond_dim
            ),
            ConditionalResBlock(
                base,
                cond_dim
            )
        ])

        self.out_norm = nn.GroupNorm(
            _groups(base),
            base
        )

        self.out = nn.Conv2d(
            base,
            3,
            3,
            1,
            1
        )

    def _blocks(
        self,
        blocks,
        x,
        cond
    ):

        for block in blocks:
            x = block(x, cond)

        return x

    def forward(
        self,
        x_t,
        t,
        source,
        identity
    ):

        h = self.in_conv(
            torch.cat(
                [x_t, source],
                dim=1
            )
        )

        cond = (
            self.time(t)
            +
            self.id_proj(
                identity.float()
            )
        )

        e1 = self._blocks(
            self.e1,
            h,
            cond
        )

        e2 = self._blocks(
            self.e2,
            self.down1(e1),
            cond
        )

        e3 = self._blocks(
            self.e3,
            self.down2(e2),
            cond
        )

        m = self._blocks(
            self.mid,
            self.down3(e3),
            cond
        )

        m = self.cross_attn(
            m,
            identity.float()
        )

        m = F.interpolate(
            m,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = self._blocks(
            self.d3,
            self.up3(m) + e3,
            cond
        )

        d3 = F.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = self._blocks(
            self.d2,
            self.up2(d3) + e2,
            cond
        )

        d2 = F.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = self._blocks(
            self.d1,
            self.up1(d2) + e1,
            cond
        )

        return self.out(
            self.out_norm(d1)
        )


# ============================================================
# Checkpoint handling
# ============================================================

def _find_checkpoint_file(root):
    """
    Find the actual PyTorch checkpoint inside a .zip directory.
    Preference:
        best.pt
        latest.pt
        final_model_weights.pt
        other .pt/.pth files
    """

    root = Path(root)

    candidates = list(root.rglob("*.pt"))
    candidates += list(root.rglob("*.pth"))

    if not candidates:
        raise FileNotFoundError(
            f"No .pt/.pth checkpoint found inside {root}"
        )

    priority = [
        "best.pt",
        "latest.pt",
        "final_model_weights.pt",
    ]

    for name in priority:
        for p in candidates:
            if p.name == name:
                return p

    return candidates[0]


def _load_checkpoint(checkpoint, device):
    """
    Load a flow checkpoint.

    The notebook's .zip files are actually PyTorch serialized
    checkpoints whose internal archive is named, for example:

        best.pt/data.pkl
        best.pt/data/0
        best.pt/data/1
        ...

    Therefore they must NOT be extracted with zipfile.

    torch.load() can read the serialized checkpoint directly.
    """

    import pathlib

    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist:\n{checkpoint}"
        )

    print(
        f"[Flow] Loading checkpoint: {checkpoint}"
    )

    # The notebook ran on Kaggle (Linux), so its "config" dict holds
    # pickled PosixPath objects. On Windows those can't be rebuilt,
    # so temporarily map them to WindowsPath while unpickling.
    _orig_posix = pathlib.PosixPath

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath

    try:

        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False
        )

    except Exception as exc:

        raise RuntimeError(
            "\nCould not load the Flow checkpoint.\n"
            f"Checkpoint: {checkpoint}\n\n"
            "This file should be the original PyTorch checkpoint "
            "exported by the training notebook "
            "(e.g. best.zip or final_model_weights.zip).\n\n"
            f"Original error:\n{exc}"
        ) from exc

    finally:

        pathlib.PosixPath = _orig_posix

    if isinstance(payload, dict):

        print(
            "[Flow] Checkpoint loaded successfully."
        )

        print(
            f"[Flow] Checkpoint keys: "
            f"{list(payload.keys())}"
        )

    else:

        print(
            "[Flow] Checkpoint loaded successfully."
        )

        print(
            f"[Flow] Checkpoint type: "
            f"{type(payload).__name__}"
        )

    return payload
# ============================================================
# Flow sampler
# ============================================================

@torch.inference_mode()
def _flow_sample(
    model,
    source,
    identity,
    noise,
    device,
    steps=4,
    solver="heun"
):

    model.eval()

    x = noise.float()

    source = source.float()

    identity = identity.float()

    # Exactly matches notebook's:
    #
    # ts=torch.linspace(0,1,steps+1,...)
    #
    ts = torch.linspace(
        0,
        1,
        steps + 1,
        device=device,
        dtype=torch.float32
    )

    for i in range(steps):

        t0 = ts[i].expand(
            x.shape[0]
        )

        t1 = ts[i + 1].expand(
            x.shape[0]
        )

        dt = (
            ts[i + 1]
            -
            ts[i]
        )

        v0 = model(
            x,
            t0,
            source,
            identity
        ).float()

        if solver.lower() == "heun":

            x_euler = x + dt * v0

            v1 = model(
                x_euler,
                t1,
                source,
                identity
            ).float()

            x = (
                x
                +
                dt
                *
                0.5
                *
                (v0 + v1)
            )

        elif solver.lower() == "euler":

            x = x + dt * v0

        else:

            raise ValueError(
                "solver must be 'euler' or 'heun'"
            )

    return x.clamp(-1, 1)


# ============================================================
# Main Pipeline
# ============================================================

class FlowPipeline:

    def __init__(
        self,
        model,
        bisenet,
        identity_model,
        device,
        sample_steps=DEFAULT_SAMPLE_STEPS,
        solver=DEFAULT_SOLVER,
        feather=24
    ):

        self.model = model
        self.bisenet = bisenet
        self.identity_model = identity_model

        self.device = device

        self.sample_steps = sample_steps
        self.solver = solver

        self.feather = feather

        self.model.eval()
        self.bisenet.eval()
        self.identity_model.eval()

    # --------------------------------------------------------
    # Face box
    # --------------------------------------------------------

    @torch.inference_mode()
    def _get_face_box(self, image):

        x = image.unsqueeze(0).to(
            self.device
        )

        mask = self.bisenet(x)

        box = mask_to_bbox(mask)[0]

        return box

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    @staticmethod
    def _crop_from_box(
        img,
        box,
        out_size=CROP_SIZE
    ):

        top, left, size = map(
            int,
            box
        )

        crop = img[
            :,
            top:top + size,
            left:left + size
        ]

        crop = F.interpolate(
            crop.unsqueeze(0),
            (out_size, out_size),
            mode="bilinear",
            align_corners=False
        )[0]

        return crop

    # --------------------------------------------------------
    # Identity embedding
    # --------------------------------------------------------

    @torch.inference_mode()
    def _get_identity_embedding(
        self,
        target,
        target_box
    ):

        crop = self._crop_from_box(
            target,
            target_box,
            out_size=160
        )

        crop = crop.unsqueeze(0).to(
            self.device
        )

        # Exactly matches training:
        #
        # x = x * 2 - 1
        # emb = embedder(x)
        # emb = F.normalize(emb, dim=1)

        crop = crop * 2 - 1

        embedding = self.identity_model(
            crop
        ).float()

        embedding = F.normalize(
            embedding,
            dim=1
        )

        return embedding

    # --------------------------------------------------------
    # Feather mask
    # --------------------------------------------------------

    def _box_mask(
        self,
        box,
        size=IMG_SIZE
    ):

        top, left, sz = map(
            int,
            box
        )

        mask = torch.zeros(
            1,
            1,
            size,
            size,
            device=self.device
        )

        mask[
            :,
            :,
            top:top + sz,
            left:left + sz
        ] = 1.0

        feather = self.feather

        k = max(
            3,
            2 * (feather // 2) + 1
        )

        if k > 1:

            mask = kornia.filters.gaussian_blur2d(
                mask,
                (k, k),
                (
                    feather / 2,
                    feather / 2
                )
            )

        return mask.clamp(0, 1)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        source,
        target,
        steps=None,
        solver=None,
        seed=None
    ):

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        source_path = Path(source)
        target_path = Path(target)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source image not found:\n{source_path}"
            )

        if not target_path.exists():
            raise FileNotFoundError(
                f"Target image not found:\n{target_path}"
            )

        # ----------------------------------------------------
        # Load images exactly like training
        # ----------------------------------------------------

        source_img = _load_rgb_tensor(
            source_path,
            IMG_SIZE
        )

        target_img = _load_rgb_tensor(
            target_path,
            IMG_SIZE
        )

        # ----------------------------------------------------
        # Detect source and target face boxes
        # ----------------------------------------------------

        source_box = self._get_face_box(
            source_img
        )

        target_box = self._get_face_box(
            target_img
        )

        # ----------------------------------------------------
        # Source face crop
        # ----------------------------------------------------

        source_crop = self._crop_from_box(
            source_img,
            source_box,
            CROP_SIZE
        )

        # Training uses [-1, 1]
        source_input = (
            source_crop * 2
            - 1
        )

        # ----------------------------------------------------
        # Target identity embedding
        # ----------------------------------------------------

        identity = self._get_identity_embedding(
            target_img,
            target_box
        )

        # ----------------------------------------------------
        # Random initial noise
        # ----------------------------------------------------

        if seed is not None:

            generator = torch.Generator(
                device=self.device
            )

            generator.manual_seed(
                int(seed)
            )

            noise = torch.randn(
                1,
                3,
                CROP_SIZE,
                CROP_SIZE,
                device=self.device,
                generator=generator
            )

        else:

            noise = torch.randn(
                1,
                3,
                CROP_SIZE,
                CROP_SIZE,
                device=self.device
            )

        source_input = source_input.unsqueeze(
            0
        ).to(self.device)

        # ----------------------------------------------------
        # Flow sampling
        # ----------------------------------------------------

        sample_steps = (
            self.sample_steps
            if steps is None
            else int(steps)
        )

        sample_solver = (
            self.solver
            if solver is None
            else solver
        )

        pred = _flow_sample(
            self.model,
            source_input,
            identity,
            noise,
            self.device,
            steps=sample_steps,
            solver=sample_solver
        )

        # ----------------------------------------------------
        # [-1,1] -> [0,1]
        # ----------------------------------------------------

        pred_crop = (
            pred[0].float() + 1
        ) / 2

        pred_crop = pred_crop.clamp(
            0,
            1
        )

        # ----------------------------------------------------
        # Put generated face back into source
        # ----------------------------------------------------

        result = source_img.clone()

        top, left, size = map(
            int,
            source_box
        )

        pred_face = F.interpolate(
            pred_crop.unsqueeze(0),
            (size, size),
            mode="bilinear",
            align_corners=False
        )[0]

        mask = self._box_mask(
            source_box
        )

        region = result[
            :,
            top:top + size,
            left:left + size
        ]

        local_mask = mask[
            :,
            :,
            top:top + size,
            left:left + size
        ][0]

        result[
            :,
            top:top + size,
            left:left + size
        ] = (
            pred_face * local_mask
            +
            region * (1 - local_mask)
        )

        # ----------------------------------------------------
        # PIL conversion
        # ----------------------------------------------------

        def tensor_to_pil(t):

            arr = (
                t.clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            return Image.fromarray(
                arr,
                mode="RGB"
            )

        output_image = tensor_to_pil(
            result
        )

        generated_face = tensor_to_pil(
            pred_crop
        )

        source_pil = tensor_to_pil(
            source_img
        )

        target_pil = tensor_to_pil(
            target_img
        )

        # ----------------------------------------------------
        # Return useful inference information
        # ----------------------------------------------------

        return {
            "image": output_image,

            # Generated 256x256 face before compositing.
            "face": generated_face,

            "source": source_pil,

            "target": target_pil,

            "source_box": source_box,

            "target_box": target_box,

            "steps": sample_steps,

            "solver": sample_solver,

            "device": str(self.device),
        }


# ============================================================
# Public loader
# ============================================================

def load_flow_pipeline(
    checkpoint,
    bisenet_checkpoint,
    device=None,
    sample_steps=DEFAULT_SAMPLE_STEPS,
    solver=DEFAULT_SOLVER,
    feather=24
):
    """
    Load a trained Conditional Rectified Flow pipeline.

    Parameters
    ----------
    checkpoint : str
        Path to:
            .zip
            .pt
            .pth

        If a ZIP is supplied, the loader searches inside it
        for best.pt, latest.pt, final_model_weights.pt, or
        another .pt/.pth file.

    bisenet_checkpoint : str
        Path to the 79999_iter.pth BiSeNet checkpoint.

    device : str, optional
        "cuda", "cpu", "mps", etc.
        Automatically selected when omitted.

    sample_steps : int
        Number of ODE integration steps.
        Notebook default = 4.

    solver : str
        "heun" or "euler".
        Notebook default = "heun".

    feather : int
        Feather radius used when compositing the generated
        face back into the source image.

    Returns
    -------
    FlowPipeline
    """

    device = _select_device(device)

    print(
        f"[Flow] Device: {device}"
    )

    # ========================================================
    # Load flow checkpoint
    # ========================================================

    payload = _load_checkpoint(
        checkpoint,
        device
    )

    # ========================================================
    # Construct exact FlowMatchingUNet
    # ========================================================

    model = FlowMatchingUNet(
        base=32,
        cond_dim=128
    )

    # --------------------------------------------------------
    # Training checkpoint:
    #
    # {
    #     "model": ...,
    #     "optimizer": ...,
    #     ...
    # }
    #
    # Final weights:
    #
    # {
    #     "model": ...,
    #     "config": ...,
    #     ...
    # }
    # --------------------------------------------------------

    if isinstance(payload, dict):

        if "model" in payload:
            state_dict = payload["model"]

        elif "state_dict" in payload:
            state_dict = payload["state_dict"]

        else:
            # Allow a raw state_dict.
            state_dict = payload

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # Remove possible DataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        raise RuntimeError(
            "Flow model checkpoint is missing "
            f"{len(missing)} parameters.\n"
            f"First missing parameters: {missing[:10]}"
        )

    if unexpected:
        print(
            f"[Flow] Warning: {len(unexpected)} "
            "unexpected checkpoint parameters."
        )

    model = model.to(device)
    model.eval()

    # ========================================================
    # Load BiSeNet
    # ========================================================

    bisenet_checkpoint = Path(
        bisenet_checkpoint
    )

    if not bisenet_checkpoint.exists():
        raise FileNotFoundError(
            "BiSeNet checkpoint not found:\n"
            f"{bisenet_checkpoint}"
        )

    print(
        "[Flow] Loading BiSeNet..."
    )

    bisenet = BiSeNet(
        n_classes=19
    )

    bisenet_state = torch.load(
        bisenet_checkpoint,
        map_location="cpu",
        weights_only=False
    )

    if (
        isinstance(bisenet_state, dict)
        and "state_dict" in bisenet_state
    ):
        bisenet_state = bisenet_state[
            "state_dict"
        ]

    cleaned_bisenet = {}

    for key, value in bisenet_state.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_bisenet[key] = value

    bisenet.load_state_dict(
        cleaned_bisenet,
        strict=True
    )

    bisenet = bisenet.to(
        device
    ).eval()

    mask_extractor = BiSeNetMaskExtractor(
        bisenet
    ).to(device)

    # ========================================================
    # Load VGGFace2 identity encoder
    # ========================================================

    print(
        "[Flow] Loading VGGFace2 identity encoder..."
    )

    identity_model = (
        InceptionResnetV1(
            pretrained="vggface2"
        )
        .to(device)
        .eval()
    )

    for p in identity_model.parameters():
        p.requires_grad_(False)

    # ========================================================
    # Construct pipeline
    # ========================================================

    pipeline = FlowPipeline(
        model=model,
        bisenet=mask_extractor,
        identity_model=identity_model,
        device=device,
        sample_steps=sample_steps,
        solver=solver,
        feather=feather
    )

    print(
        "[Flow] Pipeline ready."
    )

    return pipeline