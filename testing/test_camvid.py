"""
PIDNet Testing & Benchmark Script
Runs inference, FPS benchmarking, and segmentation visualization for PIDNet.
No command-line arguments needed: configure paths directly below.
"""

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# ⚙️ USER CONFIGURATION: Edit your paths & settings here
# ==============================================================================
MODEL_PATH     = "models/model_pidnetm/best.pt"         # Base .pt checkpoint file
IMAGE_PATH     = "data/camvid/images/0016E5_06060.png"   # Input image file to test
OUTPUT_PATH    = "output_pidnet_result.png"              # Output visualization save path
WARMUP_RUNS    = 10                                      # Number of warmup iterations
BENCHMARK_RUNS = 50                                      # Number of runs for measuring FPS
# ==============================================================================


# --- Architecture Definitions (from trained/camvid_pidnets.ipynb) ---
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=algc)
        return out


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale_process = nn.Sequential(BatchNorm(branch_planes * 4, momentum=bn_mom),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False))
        self.compression = nn.Sequential(BatchNorm(branch_planes * 5, momentum=bn_mom),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(BatchNorm(inplanes, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False))

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_ = self.scale0(x)
        scale_list = [
            F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc) + x_,
            F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc) + x_,
            F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc) + x_,
            F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc) + x_
        ]
        scale_out = self.scale_process(torch.cat(scale_list, 1))
        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), BatchNorm(mid_channels))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), BatchNorm(mid_channels))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False), BatchNorm(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        return (1 - sim_map) * x + sim_map * y


class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=11, planes=32, ppm_planes=96, head_planes=128, augment=False):
        super(PIDNet, self).__init__()
        self.augment = augment
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        self.ppm = PAPPM(planes * 16, ppm_planes, planes * 4)

        self.compression3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
        self.compression4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.pag3 = PagFM(planes * 2, planes)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
        self.diff3 = nn.Sequential(nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes, momentum=bn_mom))
        self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        self.diff4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom))
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.dfm = Light_Bag(planes * 4, planes * 4)
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def forward(self, x):
        h, w = x.shape[-2] // 8, x.shape[-1] // 8
        x = self.conv1(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        xp = self.layer3_p(x)
        xd = self.layer3_d(x)
        x = self.relu(self.layer3(x))
        xp = self.pag3(xp, self.compression3(x))
        xd = xd + F.interpolate(self.diff3(x), size=[h, w], mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = xp
        x = self.relu(self.layer4(x))
        xp = self.layer4_p(self.relu(xp))
        xd = self.layer4_d(self.relu(xd))
        xp = self.pag4(xp, self.compression4(x))
        xd = xd + F.interpolate(self.diff4(x), size=[h, w], mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = xd
        xp = self.layer5_p(self.relu(xp))
        xd = self.layer5_d(self.relu(xd))
        x = self.layer5(x)
        x = F.interpolate(self.ppm(x), size=[h, w], mode='bilinear', align_corners=algc)
        x_ = self.final_layer(self.dfm(xp, x, xd))
        if self.augment:
            return [self.seghead_p(temp_p), x_, self.seghead_d(temp_d)]
        return x_


CAMVID_CLASSES = [
    'Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence',
    'Pedestrian', 'Road', 'Sidewalk', 'SignSymbol', 'Sky', 'Tree'
]

CAMVID_COLORS = np.array([
    [0, 128, 192],   # Bicyclist
    [128, 0, 0],     # Building
    [64, 0, 128],    # Car
    [192, 192, 128], # Column_Pole
    [64, 64, 128],   # Fence
    [64, 64, 0],     # Pedestrian
    [128, 64, 128],  # Road
    [0, 0, 192],     # Sidewalk
    [192, 128, 128], # SignSymbol
    [128, 128, 128], # Sky
    [128, 128, 0]    # Tree
], dtype=np.uint8)


def load_model(checkpoint_path, device='cuda'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    cleaned = {k[6:] if k.startswith('model.') else (k[7:] if k.startswith('module.') else k): v for k, v in state.items()}

    # Auto-detect PIDNet-S vs PIDNet-M
    head_in = cleaned['final_layer.bn1.weight'].shape[0] if 'final_layer.bn1.weight' in cleaned else 128
    model_name = 'pidnet_s' if head_in == 128 else 'pidnet_m'
    planes = 64 if head_in == 256 else 32
    num_classes = cleaned['final_layer.conv2.weight'].shape[0] if 'final_layer.conv2.weight' in cleaned else 11

    print(f"Loading {model_name.upper()} (planes={planes}, num_classes={num_classes}) from {checkpoint_path}")
    model = PIDNet(m=2, n=3, num_classes=num_classes, planes=planes, augment=False)
    mdict = model.state_dict()
    matched = {k: v for k, v in cleaned.items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(matched)
    model.load_state_dict(mdict)
    model = model.to(device).eval()
    print(f"Successfully loaded {len(matched)} / {len(mdict)} tensors.")
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Model
    model = load_model(MODEL_PATH, device=device)

    # 2. Read and Preprocess Image
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found at: {IMAGE_PATH}")

    pil_img = Image.open(IMAGE_PATH).convert('RGB')
    rgb = np.array(pil_img)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((rgb / 255.0 - mean) / std).transpose((2, 0, 1))).unsqueeze(0).float().to(device)

    print(f"Image resolution: {rgb.shape[1]}x{rgb.shape[0]}")

    # 3. Warmup
    print(f"Warming up for {WARMUP_RUNS} iterations...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 4. Benchmark FPS
    print(f"Benchmarking over {BENCHMARK_RUNS} iterations...")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            _ = model(tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    tot = t1 - t0
    fps = BENCHMARK_RUNS / tot
    latency = (tot / BENCHMARK_RUNS) * 1000

    print("=" * 45)
    print(f"⚡ BENCHMARK RESULTS:")
    print(f"   Device:         {device.type.upper()}")
    print(f"   Avg Latency:    {latency:.2f} ms")
    print(f"   Inference Speed: {fps:.2f} FPS")
    print("=" * 45)

    # 5. Inference & Segmentation Output
    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[1]
        out = F.interpolate(out, size=rgb.shape[:2], mode='bilinear', align_corners=False)
        pred = out.argmax(dim=1)[0].cpu().numpy()

    color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cid, col in enumerate(CAMVID_COLORS):
        color_mask[pred == cid] = col

    blended = (0.55 * rgb + 0.45 * color_mask).astype(np.uint8)

    unique_classes = np.unique(pred)
    print(f"Detected {len(unique_classes)} classes:")
    for c in unique_classes:
        if c < len(CAMVID_CLASSES):
            print(f"   • {CAMVID_CLASSES[c]}")

    # 6. Visualization & Save
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].imshow(rgb)
    ax[0].set_title(f"Input Image\n({os.path.basename(IMAGE_PATH)})", fontsize=13)
    ax[0].axis('off')

    ax[1].imshow(color_mask)
    ax[1].set_title(f"Segmentation Mask\n(PIDNet: {fps:.1f} FPS | {latency:.1f} ms)", fontsize=13)
    ax[1].axis('off')

    ax[2].imshow(blended)
    ax[2].set_title("Overlay (Image + Mask)", fontsize=13)
    ax[2].axis('off')

    legend_patches = [
        mpatches.Patch(color=CAMVID_COLORS[c] / 255.0, label=CAMVID_CLASSES[c])
        for c in unique_classes if c < len(CAMVID_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(6, len(legend_patches)),
               bbox_to_anchor=(0.5, -0.06), fontsize=11, frameon=True)

    plt.tight_layout()
    if OUTPUT_PATH:
        plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == '__main__':
    main()
