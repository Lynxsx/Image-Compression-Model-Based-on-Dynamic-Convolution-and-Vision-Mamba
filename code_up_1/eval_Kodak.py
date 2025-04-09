import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from compressai.models import Mamba_Framework
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import io

warnings.filterwarnings("ignore")

print(torch.cuda.is_available())

def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim_T(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main():
    checkpoint = "./path"
    data = "./Kodak24/"
    cuda = True
    p = 256
    path = data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = Mamba_Framework()
    net = net.to(device)
    net.eval()
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    encode_time = 0.0
    decode_time = 0.0

    dictory = {}
    if checkpoint:  # load from previous checkpoint
        print("Loading", checkpoint)

        checkpoint = torch.load(checkpoint, map_location=device)
        checkpoint = checkpoint['state_dict']
        for k, v in checkpoint.items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory,strict=True)

        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(x_padded)
                if cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)

                rec_save = out_net["x_hat"].squeeze(0).permute(1,2,0).cpu().numpy()
                r_image = Image.fromarray((rec_save * 255).astype(np.uint8))
                r_image.save('./path')

                origin_img = io.imread(img_path)
                rec_img = io.imread('./path')


                SSIM_tmp = structural_similarity(origin_img, rec_img, multichannel=True, channel_axis=-1)

                encode_time_s = out_net["encode time"]
                decode_time_s = out_net["decode time"]
                print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
                print(f'MS-SSIM: {SSIM_tmp:.2f}dB')
                print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
                PSNR += compute_psnr(x, out_net["x_hat"])
                MS_SSIM += SSIM_tmp
                Bit_rate += compute_bpp(out_net)


    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    total_enc_time = encode_time / count
    total_dec_time = decode_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')
    print(f'average_encode_time: {total_enc_time:.3f} ms')
    print(f'average_decode_time: {total_dec_time:.3f} ms')


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
    