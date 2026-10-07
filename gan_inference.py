# gan_inference.py

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia

from facenet_pytorch import InceptionResnetV1


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = 512
CROP_SIZE = 256
STYLE_DIM = 512

FACE_CLASS_IDS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]


# ============================================================
# BiSeNet
# Exact architecture used in the training notebook
# ============================================================

class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_chan)

        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=3,
            stride=1,
            padding=1,
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
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_chan)
            )

    def forward(self, x):

        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))

        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)

        return out


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
                stride=1
            )
        )

    return nn.Sequential(*layers)


class Resnet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = create_layer_basic(
            64,
            64,
            bnum=2,
            stride=1
        )

        self.layer2 = create_layer_basic(
            64,
            128,
            bnum=2,
            stride=2
        )

        self.layer3 = create_layer_basic(
            128,
            256,
            bnum=2,
            stride=2
        )

        self.layer4 = create_layer_basic(
            256,
            512,
            bnum=2,
            stride=2
        )

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

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
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


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
            mid_chan,
            ks=3,
            stride=1,
            padding=1
        )

        self.conv_out = nn.Conv2d(
            mid_chan,
            n_classes,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.conv_out(x)

        return x


class AttentionRefinementModule(nn.Module):

    def __init__(
        self,
        in_chan,
        out_chan
    ):
        super().__init__()

        self.conv = ConvBNReLU(
            in_chan,
            out_chan,
            ks=3,
            stride=1,
            padding=1
        )

        self.conv_atten = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=1,
            bias=False
        )

        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):

        feat = self.conv(x)

        atten = F.avg_pool2d(
            feat,
            feat.size()[2:]
        )

        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)

        out = torch.mul(feat, atten)

        return out


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
            128,
            ks=3,
            stride=1,
            padding=1
        )

        self.conv_head16 = ConvBNReLU(
            128,
            128,
            ks=3,
            stride=1,
            padding=1
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

        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(
            feat32,
            feat32.size()[2:]
        )

        avg = self.conv_avg(avg)

        avg_up = F.interpolate(
            avg,
            (H32, W32),
            mode="nearest"
        )

        feat32_arm = self.arm32(feat32)

        feat32_sum = (
            feat32_arm +
            avg_up
        )

        feat32_up = F.interpolate(
            feat32_sum,
            (H16, W16),
            mode="nearest"
        )

        feat32_up = self.conv_head32(
            feat32_up
        )

        feat16_arm = self.arm16(feat16)

        feat16_sum = (
            feat16_arm +
            feat32_up
        )

        feat16_up = F.interpolate(
            feat16_sum,
            (H8, W8),
            mode="nearest"
        )

        feat16_up = self.conv_head16(
            feat16_up
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
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            out_chan // 4,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):

        fcat = torch.cat(
            [fsp, fcp],
            dim=1
        )

        feat = self.convblk(fcat)

        atten = F.avg_pool2d(
            feat,
            feat.size()[2:]
        )

        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)

        feat_atten = torch.mul(
            feat,
            atten
        )

        feat_out = feat_atten + feat

        return feat_out


class BiSeNet(nn.Module):

    def __init__(
        self,
        n_classes=19
    ):
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

        H, W = x.size()[2:]

        feat_res8, feat_cp8, feat_cp16 = self.cp(x)

        feat_sp = feat_res8

        feat_fuse = self.ffm(
            feat_sp,
            feat_cp8
        )

        feat_out = self.conv_out(
            feat_fuse
        )

        feat_out16 = self.conv_out16(
            feat_cp8
        )

        feat_out32 = self.conv_out32(
            feat_cp16
        )

        feat_out = F.interpolate(
            feat_out,
            (H, W),
            mode="bilinear",
            align_corners=True
        )

        feat_out16 = F.interpolate(
            feat_out16,
            (H, W),
            mode="bilinear",
            align_corners=True
        )

        feat_out32 = F.interpolate(
            feat_out32,
            (H, W),
            mode="bilinear",
            align_corners=True
        )

        return (
            feat_out,
            feat_out16,
            feat_out32
        )


# ============================================================
# Face mask / cropping
# ============================================================

class BiSeNetMaskExtractor(nn.Module):

    def __init__(
        self,
        bisenet,
        dilate_px=6,
        feather_px=9
    ):
        super().__init__()

        self.bisenet = bisenet

        self.bisenet.eval()

        for p in self.bisenet.parameters():
            p.requires_grad_(False)

        self.dilate_px = dilate_px
        self.feather_px = feather_px

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

    @torch.no_grad()
    def forward(self, img):

        B, C, H, W = img.shape

        img_512 = F.interpolate(
            img,
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )

        img_norm = (
            img_512 - self.mean
        ) / self.std

        logits, _, _ = self.bisenet(
            img_norm
        )

        parsing = logits.argmax(
            dim=1,
            keepdim=True
        )

        face_mask = torch.zeros_like(
            parsing,
            dtype=torch.float32
        )

        for cls_id in FACE_CLASS_IDS:
            face_mask = (
                face_mask +
                (parsing == cls_id).float()
            )

        face_mask = face_mask.clamp(
            0,
            1
        )

        face_mask = F.interpolate(
            face_mask,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        if self.dilate_px > 0:

            k = (
                self.dilate_px * 2 +
                1
            )

            kernel = torch.ones(
                (k, k),
                device=img.device
            )

            face_mask = kornia.morphology.dilation(
                face_mask,
                kernel
            )

        if self.feather_px > 0:

            k = (
                self.feather_px * 2 +
                1
            )

            face_mask = kornia.filters.gaussian_blur2d(
                face_mask,
                (k, k),
                (
                    self.feather_px / 2.0,
                    self.feather_px / 2.0
                )
            )

        return face_mask.clamp(
            0,
            1
        )


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

            boxes.append(
                (
                    H // 2 - size // 2,
                    W // 2 - size // 2,
                    size
                )
            )

            continue

        y0 = ys.min().item()
        y1 = ys.max().item()

        x0 = xs.min().item()
        x1 = xs.max().item()

        h = y1 - y0
        w = x1 - x0

        size = int(
            max(h, w) *
            (1 + margin_ratio)
        )

        size = max(
            size,
            8
        )

        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2

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

        boxes.append(
            (
                top,
                left,
                size
            )
        )

    return boxes


def crop_faces(
    img,
    boxes,
    out_size=256
):

    crops = []

    for b, (top, left, size) in enumerate(boxes):

        crop = img[
            b:b + 1,
            :,
            top:top + size,
            left:left + size
        ]

        crop = F.interpolate(
            crop,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False
        )

        crops.append(crop)

    return torch.cat(
        crops,
        dim=0
    )


def paste_faces_back(
    full_img,
    face_crops,
    boxes
):

    out = full_img.clone()

    for b, (top, left, size) in enumerate(boxes):

        resized = F.interpolate(
            face_crops[b:b + 1],
            size=(size, size),
            mode="bilinear",
            align_corners=False
        )

        out[
            b:b + 1,
            :,
            top:top + size,
            left:left + size
        ] = resized

    return out


# ============================================================
# Generator
# Exact architecture used during training
# ============================================================

class AdaIN(nn.Module):

    def __init__(
        self,
        num_features,
        style_dim
    ):
        super().__init__()

        self.norm = nn.InstanceNorm2d(
            num_features,
            affine=False
        )

        self.style_fc = nn.Linear(
            style_dim,
            num_features * 2
        )

    def forward(
        self,
        x,
        style
    ):

        h = self.style_fc(style)

        gamma, beta = h.chunk(
            2,
            dim=1
        )

        gamma = gamma.unsqueeze(
            -1
        ).unsqueeze(-1)

        beta = beta.unsqueeze(
            -1
        ).unsqueeze(-1)

        return (
            (1 + gamma) *
            self.norm(x)
            + beta
        )


class ResBlockAdaIN(nn.Module):

    def __init__(
        self,
        channels,
        style_dim
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1
        )

        self.adain1 = AdaIN(
            channels,
            style_dim
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1
        )

        self.adain2 = AdaIN(
            channels,
            style_dim
        )

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(
        self,
        x,
        style
    ):

        residual = x

        x = self.conv1(x)
        x = self.adain1(
            x,
            style
        )
        x = self.act(x)

        x = self.conv2(x)
        x = self.adain2(
            x,
            style
        )

        return residual + x


class FaceSwapGenerator(nn.Module):

    def __init__(
        self,
        style_dim=512,
        base_ch=32
    ):
        super().__init__()

        self.enc = nn.Sequential(

            nn.Conv2d(
                3,
                base_ch,
                7,
                1,
                3
            ),

            nn.InstanceNorm2d(
                base_ch
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                base_ch,
                base_ch * 2,
                4,
                2,
                1
            ),

            nn.InstanceNorm2d(
                base_ch * 2
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                base_ch * 2,
                base_ch * 4,
                4,
                2,
                1
            ),

            nn.InstanceNorm2d(
                base_ch * 4
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                base_ch * 4,
                base_ch * 8,
                4,
                2,
                1
            ),

            nn.InstanceNorm2d(
                base_ch * 8
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        self.bottleneck = nn.ModuleList([
            ResBlockAdaIN(
                base_ch * 8,
                style_dim
            )
            for _ in range(6)
        ])

        self.dec_up1 = nn.ConvTranspose2d(
            base_ch * 8,
            base_ch * 4,
            4,
            2,
            1
        )

        self.dec_norm1 = nn.InstanceNorm2d(
            base_ch * 4
        )

        self.dec_up2 = nn.ConvTranspose2d(
            base_ch * 4,
            base_ch * 2,
            4,
            2,
            1
        )

        self.dec_norm2 = nn.InstanceNorm2d(
            base_ch * 2
        )

        self.dec_up3 = nn.ConvTranspose2d(
            base_ch * 2,
            base_ch,
            4,
            2,
            1
        )

        self.dec_norm3 = nn.InstanceNorm2d(
            base_ch
        )

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(
                base_ch,
                3,
                7,
                1,
                3
            ),
            nn.Tanh()
        )

    def forward(
        self,
        source_crop,
        identity_embed
    ):

        feat = self.enc(
            source_crop
        )

        for block in self.bottleneck:
            feat = block(
                feat,
                identity_embed
            )

        x = self.dec_up1(feat)
        x = self.dec_norm1(x)
        x = self.act(x)

        x = self.dec_up2(x)
        x = self.dec_norm2(x)
        x = self.act(x)

        x = self.dec_up3(x)
        x = self.dec_norm3(x)
        x = self.act(x)

        return self.to_rgb(x)


# ============================================================
# Identity model
# ============================================================

class IdentityEmbedder(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = InceptionResnetV1(
            pretrained="vggface2"
        ).eval()

        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        x,
        allow_grad=False
    ):

        x160 = F.interpolate(
            x,
            size=(160, 160),
            mode="bilinear",
            align_corners=False
        )

        return self.net(
            x160
        )


# ============================================================
# Inference pipeline
# ============================================================

class FaceSwapInferencePipeline(nn.Module):

    def __init__(
        self,
        bisenet,
        identity_embedder,
        device
    ):
        super().__init__()

        self.device = device

        self.mask_extractor = BiSeNetMaskExtractor(
            bisenet
        )

        self.identity_embedder = identity_embedder

        self.generator = FaceSwapGenerator(
            style_dim=STYLE_DIM
        )

        self.crop_size = CROP_SIZE

    @torch.inference_mode()
    def forward(
        self,
        source_img,
        target_img
    ):

        # ----------------------------------------------------
        # Both images are already:
        #
        #   B x 3 x 512 x 512
        #
        # in [0, 1]
        # ----------------------------------------------------

        source_mask = self.mask_extractor(
            source_img
        )

        source_boxes = mask_to_bbox(
            source_mask
        )

        source_crop = crop_faces(
            source_img,
            source_boxes,
            self.crop_size
        )

        # ----------------------------------------------------
        # Target face detection/cropping
        #
        # This reproduces get_target_face_crop() from
        # the training notebook.
        # ----------------------------------------------------

        target_mask = self.mask_extractor(
            target_img
        )

        target_boxes = mask_to_bbox(
            target_mask
        )

        target_crop = crop_faces(
            target_img,
            target_boxes,
            self.crop_size
        )

        # ----------------------------------------------------
        # Normalize to [-1, 1]
        # Exactly as training
        # ----------------------------------------------------

        source_crop_norm = (
            source_crop * 2.0 - 1.0
        )

        target_crop_norm = (
            target_crop * 2.0 - 1.0
        )

        # ----------------------------------------------------
        # Target identity embedding
        # ----------------------------------------------------

        identity_embed = self.identity_embedder(
            target_crop_norm
        )

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        gen_face = self.generator(
            source_crop_norm,
            identity_embed
        )

        # Generator output is [-1, 1]
        gen_face_01 = (
            gen_face + 1.0
        ) / 2.0

        gen_face_01 = gen_face_01.clamp(
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Put generated face back into source
        # ----------------------------------------------------

        full_res_with_face = paste_faces_back(
            source_img,
            gen_face_01,
            source_boxes
        )

        # ----------------------------------------------------
        # Soft mask composite
        #
        # Exactly:
        #
        # composite =
        #     mask * generated
        #     +
        #     (1-mask) * source
        # ----------------------------------------------------

        composite = (
            source_mask * full_res_with_face
            +
            (1.0 - source_mask) * source_img
        )

        composite = composite.clamp(
            0.0,
            1.0
        )

        return {
            "image": composite,
            "face": gen_face_01,
            "mask": source_mask,
            "source_crop": source_crop,
            "target_crop": target_crop,
        }


# ============================================================
# Input preprocessing
# ============================================================

_IMAGE_TRANSFORM = T.Compose([
    T.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    T.ToTensor(),
])


def _load_image(
    image,
    device
):

    # --------------------------------------------------------
    # Already a torch Tensor
    # --------------------------------------------------------

    if torch.is_tensor(image):

        x = image.detach()

        if x.ndim == 3:
            x = x.unsqueeze(0)

        if x.ndim != 4:
            raise ValueError(
                "Tensor input must have shape "
                "(3,H,W) or (B,3,H,W)."
            )

        if x.shape[1] != 3:
            raise ValueError(
                "Tensor input must have 3 channels."
            )

        x = x.float()

        # The training pipeline expects [0,1].
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        x = x.clamp(
            0.0,
            1.0
        )

        x = F.interpolate(
            x,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return x.to(device)

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    if isinstance(
        image,
        np.ndarray
    ):

        if image.ndim == 3:
            pil = Image.fromarray(
                image.astype(np.uint8)
            )

            x = _IMAGE_TRANSFORM(
                pil
            ).unsqueeze(0)

        else:
            raise ValueError(
                "NumPy input must have shape (H,W,C)."
            )

        return x.to(device)

    # --------------------------------------------------------
    # PIL Image
    # --------------------------------------------------------

    if isinstance(
        image,
        Image.Image
    ):

        image = image.convert(
            "RGB"
        )

        x = _IMAGE_TRANSFORM(
            image
        ).unsqueeze(0)

        return x.to(device)

    # --------------------------------------------------------
    # File path
    # --------------------------------------------------------

    if isinstance(
        image,
        (str, os.PathLike)
    ):

        image = Image.open(
            image
        ).convert("RGB")

        x = _IMAGE_TRANSFORM(
            image
        ).unsqueeze(0)

        return x.to(device)

    raise TypeError(
        "Unsupported image input type: "
        f"{type(image)}"
    )


# ============================================================
# Checkpoint loader
# ============================================================

def _is_pytorch_archive(
    path
):

    try:

        with zipfile.ZipFile(path, "r") as zf:

            names = zf.namelist()

            if not names:
                return False

            # PyTorch's zip serialization normally contains
            # data.pkl / version / byteorder etc.
            return any(
                name.endswith("data.pkl")
                for name in names
            )

    except zipfile.BadZipFile:
        return False


def _load_checkpoint(
    checkpoint_path,
    device
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # First attempt:
    #
    # A .pt file, or a PyTorch archive whose extension happens
    # to be .zip.
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        if isinstance(
            checkpoint,
            dict
        ):

            return checkpoint

    except Exception:
        pass

    # --------------------------------------------------------
    # Normal ZIP containing a .pt checkpoint
    # --------------------------------------------------------

    if zipfile.is_zipfile(
        checkpoint_path
    ):

        temp_dir = tempfile.mkdtemp(
            prefix="gan_checkpoint_"
        )

        try:

            with zipfile.ZipFile(
                checkpoint_path,
                "r"
            ) as zf:

                zf.extractall(
                    temp_dir
                )

            extracted = list(
                Path(temp_dir).rglob("*")
            )

            # Look for a normal .pt/.pth file.
            candidates = [
                p
                for p in extracted
                if p.is_file()
                and p.suffix.lower()
                in [".pt", ".pth"]
            ]

            if candidates:

                checkpoint = torch.load(
                    candidates[0],
                    map_location=device,
                    weights_only=False
                )

                return checkpoint

            # ------------------------------------------------
            # The Kaggle notebook's special case:
            #
            # ZIP contains the internal contents of a
            # PyTorch checkpoint:
            #
            # ckpt_step90000/
            #     data.pkl
            #     byteorder
            #     version
            #     data/
            #     ...
            #
            # Reconstruct a valid PyTorch archive.
            # ------------------------------------------------

            files = [
                p
                for p in extracted
                if p.is_file()
            ]

            if not files:
                raise RuntimeError(
                    "ZIP file contains no files."
                )

            # Find a common root if one exists.
            relative_paths = [
                p.relative_to(temp_dir)
                for p in files
            ]

            # Determine whether everything is already inside
            # a single directory.
            first_parts = {
                p.parts[0]
                for p in relative_paths
                if len(p.parts) > 1
            }

            if len(first_parts) == 1:

                root_name = next(
                    iter(first_parts)
                )

                source_root = (
                    Path(temp_dir) /
                    root_name
                )

                archive_root = root_name

            else:

                source_root = Path(
                    temp_dir
                )

                archive_root = (
                    checkpoint_path.stem
                )

            reconstructed = Path(
                temp_dir
            ) / "reconstructed.pt"

            with zipfile.ZipFile(
                reconstructed,
                mode="w",
                compression=zipfile.ZIP_STORED
            ) as zf:

                for file_path in source_root.rglob("*"):

                    if not file_path.is_file():
                        continue

                    relative = (
                        file_path.relative_to(
                            source_root
                        )
                    )

                    archive_path = (
                        Path(archive_root) /
                        relative
                    )

                    zf.write(
                        file_path,
                        arcname=str(
                            archive_path
                        )
                    )

            checkpoint = torch.load(
                reconstructed,
                map_location=device,
                weights_only=False
            )

            return checkpoint

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

    raise RuntimeError(
        "Could not load checkpoint.\n"
        f"Path: {checkpoint_path}\n"
        "The file is neither a valid PyTorch checkpoint "
        "nor a recognized ZIP checkpoint."
    )


# ============================================================
# PUBLIC API
# ============================================================

def load_GAN_pipeline(
    checkpoint,
    bisenet_checkpoint,
    device=None
):
    """
    Load the trained face-swap GAN and prepare it for inference.

    Parameters
    ----------
    checkpoint : str or pathlib.Path
        Path to one of your GAN checkpoint files.

        This can be:
            - .pt
            - .pth
            - .zip containing a .pt
            - the special ZIP representation produced from
              the PyTorch checkpoint by Kaggle.

    bisenet_checkpoint : str or pathlib.Path
        Path to the 79999_iter.pth BiSeNet checkpoint used
        during training.

    device : str or torch.device, optional
        "cuda", "cuda:0", or "cpu".
        If None, CUDA is used when available.

    Returns
    -------
    FaceSwapGAN
        Callable inference object.

    Example
    -------
    gan = load_GAN_pipeline(
        checkpoint=r"D:\\models\\ckpt_epoch11.zip",
        bisenet_checkpoint=r"D:\\models\\79999_iter.pth"
    )

    result = gan(
        source=r"D:\\images\\source.jpg",
        target=r"D:\\images\\target.jpg"
    )

    result["image"]
        -> torch.Tensor, shape (1,3,512,512), [0,1]
    """

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device
        )

    # --------------------------------------------------------
    # BiSeNet
    # --------------------------------------------------------

    bisenet = BiSeNet(
        n_classes=19
    )

    bisenet_state = torch.load(
        bisenet_checkpoint,
        map_location="cpu",
        weights_only=False
    )

    # Some segmentation checkpoints may be wrapped.
    if isinstance(
        bisenet_state,
        dict
    ) and "state_dict" in bisenet_state:

        bisenet_state = (
            bisenet_state["state_dict"]
        )

    # Remove possible "module." prefix.
    if isinstance(
        bisenet_state,
        dict
    ):

        cleaned_state = {}

        for key, value in bisenet_state.items():

            if key.startswith(
                "module."
            ):
                key = key[7:]

            cleaned_state[key] = value

        bisenet_state = cleaned_state

    bisenet.load_state_dict(
        bisenet_state,
        strict=True
    )

    bisenet.eval()

    for p in bisenet.parameters():
        p.requires_grad_(False)

    bisenet = bisenet.to(
        device
    )

    # --------------------------------------------------------
    # Identity model
    # --------------------------------------------------------

    identity_embedder = IdentityEmbedder()

    identity_embedder = (
        identity_embedder.to(device)
    )

    identity_embedder.eval()

    # --------------------------------------------------------
    # Build inference pipeline
    # --------------------------------------------------------

    pipeline = FaceSwapInferencePipeline(
        bisenet=bisenet,
        identity_embedder=identity_embedder,
        device=device
    )

    # --------------------------------------------------------
    # Load GAN checkpoint
    # --------------------------------------------------------

    checkpoint_data = _load_checkpoint(
        checkpoint,
        device
    )

    if not isinstance(
        checkpoint_data,
        dict
    ):
        raise RuntimeError(
            "GAN checkpoint did not contain a dictionary."
        )

    if "generator" not in checkpoint_data:

        raise KeyError(
            "The checkpoint does not contain "
            "'generator'.\n\n"
            f"Available keys: "
            f"{list(checkpoint_data.keys())}"
        )

    # --------------------------------------------------------
    # Restore ONLY the generator.
    #
    # discriminator / optimizers / global_step are irrelevant
    # for inference.
    # --------------------------------------------------------

    pipeline.generator.load_state_dict(
        checkpoint_data["generator"],
        strict=True
    )

    pipeline.generator.eval()

    pipeline.eval()

    # --------------------------------------------------------
    # Keep everything frozen
    # --------------------------------------------------------

    for p in pipeline.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # Store useful metadata
    # --------------------------------------------------------

    pipeline.device = device

    pipeline.checkpoint_path = str(
        checkpoint
    )

    pipeline.bisenet_checkpoint = str(
        bisenet_checkpoint
    )

    if "global_step" in checkpoint_data:
        pipeline.training_step = (
            checkpoint_data["global_step"]
        )
    else:
        pipeline.training_step = None

    # --------------------------------------------------------
    # User-facing callable wrapper
    # --------------------------------------------------------

    return _InferenceWrapper(
        pipeline
    )


# ============================================================
# Simple callable wrapper
# ============================================================

class _InferenceWrapper:

    def __init__(
        self,
        pipeline
    ):

        self.pipeline = pipeline
        self.device = pipeline.device

    @torch.inference_mode()
    def __call__(
        self,
        source,
        target
    ):
        """
        Run face swapping.

        Parameters
        ----------
        source :
            Path, PIL Image, NumPy array, or Tensor.

        target :
            Path, PIL Image, NumPy array, or Tensor.

        Returns
        -------
        dict

        Keys:

            image
                Final composited image.
                Tensor: (1,3,512,512), [0,1]

            face
                Generated face crop.
                Tensor: (1,3,256,256), [0,1]

            mask
                Source face mask.
                Tensor: (1,1,512,512), [0,1]

            source_crop
                Source face crop.

            target_crop
                Target face crop.
        """

        source_tensor = _load_image(
            source,
            self.device
        )

        target_tensor = _load_image(
            target,
            self.device
        )

        if (
            source_tensor.shape[0] !=
            target_tensor.shape[0]
        ):

            if source_tensor.shape[0] == 1:
                source_tensor = source_tensor.expand(
                    target_tensor.shape[0],
                    -1,
                    -1,
                    -1
                )

            elif target_tensor.shape[0] == 1:
                target_tensor = target_tensor.expand(
                    source_tensor.shape[0],
                    -1,
                    -1,
                    -1
                )

            else:
                raise ValueError(
                    "Source and target batch sizes "
                    "must match."
                )

        result = self.pipeline(
            source_tensor,
            target_tensor
        )

        return result

    def __repr__(self):

        step = self.pipeline.training_step

        return (
            "FaceSwapGAN(\n"
            f"  device={self.device},\n"
            f"  checkpoint={self.pipeline.checkpoint_path!r},\n"
            f"  bisenet={self.pipeline.bisenet_checkpoint!r},\n"
            f"  training_step={step},\n"
            f"  input_size={IMG_SIZE},\n"
            f"  face_size={CROP_SIZE}\n"
            ")"
        )