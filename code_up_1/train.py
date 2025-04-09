import argparse
import math
import os
import random
import shutil

from utils.Meter import AverageMeterTEST, AverageMeterTRAIN
import sys
import time
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import numpy as np
from compressai.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.models import Mamba_Framework
from pytorch_msssim import ms_ssim


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["ssim"] = ms_ssim(output["x_hat"], target, data_range=1.)
        out["loss"] = self.lmbda * 255 * 255 * out["mse_loss"] + out["bpp_loss"]
        return out


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_learning_rate,
    )
    return optimizer, aux_optimizer


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeterTEST()
    bpp_loss = AverageMeterTEST()
    mse_loss = AverageMeterTEST()
    ssim_loss = AverageMeterTEST()
    aux_loss = AverageMeterTEST()
    bpp_z_loss = AverageMeterTEST()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            N, _, H, W = d.size()
            num_pixels = N * H * W
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            bpp_z = (torch.log(out_net["likelihoods"]['z']).sum() / (-math.log(2) * num_pixels))

            loss.update(out_criterion["loss"])
            mse_losss = out_criterion["mse_loss"]
            ssim_losss = out_criterion['ssim']
            psnr = 10 * (torch.log(1 / mse_losss) / np.log(10))

            mse_loss.update(psnr)
            bpp_z_loss.update(bpp_z)
            ssim_loss.update(ssim_losss)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE (PSNR): {mse_loss.avg :.3f} |"
        f"\tSSIM loss: {ssim_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tBpp z loss: {bpp_z_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg

elapsed, data_times, losses, psnrs, bpps, bpp_ys, bpp_zs, mse_losses, aux_losses = [AverageMeterTRAIN(2000) for _ in
                                                                                    range(9)]

def save_checkpoint(state, is_best, filename="checkpoint.pth", output_dir="./path/"):
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_checkpoint_path = os.path.join(output_dir, "checkpoint_best_loss.pth")
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

def main():
    seed = 0.6
    cuda = True
    patch_size = (256, 256)
    dataset = "/path"
    num_workers = 8
    lmbda = 0.0067
    test = False
    epochs = 500
    clip_max_norm = 1.0
    model_path = './path'
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(dataset, split="test", transform=test_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = Mamba_Framework()
    net = net.to(device)

    if os.path.exists(model_path):
        print("Loading previous model param:")
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))
    else:
        last_epoch = 0
        best_loss = float("inf")

    optimizer, aux_optimizer = configure_optimizers(net, learning_rate=1e-4, aux_learning_rate=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=10)
    criterion = RateDistortionLoss(lmbda=lmbda)

    if cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if test:
        test_epoch(0, test_dataloader, net, criterion)
        exit(-1)

    for epoch in range(last_epoch+1, epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        net.train()
        device = next(net.parameters()).device
        for i, d in enumerate(train_dataloader):
            d = d.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            out_net = net(d)
            out_criterion = criterion(out_net, d)
            out_criterion["loss"].backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_max_norm)
            optimizer.step()
            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if i % 2 == 0:
                mse_loss = out_criterion['mse_loss']
                if mse_loss.item() > 0:
                    psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                    psnrs.update(psnr.item())
                else:
                    psnrs.update(100)
                losses.update(out_criterion['loss'].item())
                bpps.update(out_criterion['bpp_loss'].item())
                mse_losses.update(mse_loss.item())
                aux_losses.update(aux_loss.item())

            if i % 10 == 0:
                print(' | '.join([
                    f'Epoch {epoch}',
                    f"{i * len(d)}/{len(train_dataloader.dataset)}",
                    f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'aux_losses Loss {aux_losses.val:.3f} ({aux_losses.avg:.3f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                    f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',

                ]))

        net.eval()
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)
        net.train()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                },

            is_best,
        )



if __name__ == "__main__":
    main()
