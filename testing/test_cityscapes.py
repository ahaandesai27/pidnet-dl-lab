import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Config -----------------------------------------------------------------
WEIGHTS = './Presentation/models/model_cityscapes_pidnets/best.pt'
DEFAULT_INPUT_DIR = os.path.join('Presentation', 'Roboflow', 'images')
OUTPUT_DIR = os.path.join('Presentation', 'Roboflow', 'output')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cityscapes 19-class palette (RGB)
COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)


# ------------------------------------------------------------------------------
# 2. PIDNet Architecture Definition
# ------------------------------------------------------------------------------
bn_mom = 0.1
algc = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
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
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
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
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=bn_mom)
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
    def __init__(self, inplanes, branch_planes, outplanes):
        super(PAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale_process = nn.Sequential(nn.BatchNorm2d(branch_planes * 4, momentum=bn_mom),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False))
        self.compression = nn.Sequential(nn.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(nn.BatchNorm2d(inplanes, momentum=bn_mom),
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
    def __init__(self, in_channels, out_channels):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mid_channels))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mid_channels))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels))
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
    """
    PIDNet (Proportional-Integral-Derivative Network)
    For PIDNet-S: m=2, n=3, planes=32, ppm_planes=96, head_planes=128
    """
    def __init__(self, m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=False):
        super(PIDNet, self).__init__()
        self.augment = augment
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        self.ppm = PAPPM(planes * 16, ppm_planes, planes * 4)

        self.compression3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False), nn.BatchNorm2d(planes * 2, momentum=bn_mom))
        self.compression4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False), nn.BatchNorm2d(planes * 2, momentum=bn_mom))
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.pag3 = PagFM(planes * 2, planes)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
        self.diff3 = nn.Sequential(nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(planes, momentum=bn_mom))
        self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        self.diff4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(planes * 2, momentum=bn_mom))
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.dfm = Light_Bag(planes * 4, planes * 4)
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom))
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



def load_pidnet(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)

    cleaned = {}
    for k, v in state.items():
        k = k[7:] if k.startswith('module.') else k
        k = k[6:] if k.startswith('model.') else k
        cleaned[k] = v

    head_in = cleaned['final_layer.bn1.weight'].shape[0] if 'final_layer.bn1.weight' in cleaned else 128
    planes = 64 if head_in == 256 else 32
    num_classes = cleaned['final_layer.conv2.weight'].shape[0] if 'final_layer.conv2.weight' in cleaned else 19

    model = PIDNet(m=2, n=3, num_classes=num_classes, planes=planes, augment=False)
    mdict = model.state_dict()
    mdict.update({k: v for k, v in cleaned.items() if k in mdict and v.shape == mdict[k].shape})
    model.load_state_dict(mdict)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img_rgb, device):
    """Pad to a multiple of 32, run the model, crop back to the original size."""
    h, w = img_rgb.shape[:2]
    pad_h, pad_w = (32 - h % 32) % 32, (32 - w % 32) % 32
    if pad_h or pad_w:
        img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    x = ((img_rgb / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)

    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[1]
    out = F.interpolate(out, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
    return out[:, :, :h, :w].argmax(dim=1)[0].cpu().numpy()


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_pidnet(WEIGHTS, device)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        print(f"No images found in {input_dir}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running on {len(files)} images from {input_dir} -> {OUTPUT_DIR}")

    for i, name in enumerate(files, 1):
        img_bgr = cv2.imread(os.path.join(input_dir, name))
        if img_bgr is None:
            print(f"  skipped (unreadable): {name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = predict(model, img_rgb, device)

        mask_rgb = COLORS[np.clip(pred, 0, len(COLORS) - 1)]
        overlay = (0.55 * img_rgb + 0.45 * mask_rgb).astype(np.uint8)

        # original | mask | overlay
        combined = np.concatenate([img_rgb, overlay], axis=1)
        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(name)[0]}_pred.png")
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  [{i}/{len(files)}] {name}")

    print("Done.")


if __name__ == '__main__':
    main()