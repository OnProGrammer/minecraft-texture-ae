import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import InterpolationMode

import imageio.v2 as imageio
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_png_paths(root):
    return [p for p in sorted(root.rglob("*.png")) if not p.name.startswith(".")]


def upscale_raw(raw_dir, upscaled_dir, image_size):
    if upscaled_dir.exists():
        shutil.rmtree(upscaled_dir, ignore_errors=True)
    upscaled_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = collect_png_paths(raw_dir)
    upscaled_paths = []

    for src_path in raw_paths:
        dst_path = upscaled_dir / src_path.relative_to(raw_dir)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src_path).convert("RGB")
        if img.size != (16, 16):
            img = img.resize((16, 16), Image.NEAREST)
        img = img.resize((image_size, image_size), Image.NEAREST)
        img.save(dst_path)
        upscaled_paths.append(dst_path)

    return upscaled_paths


class BlocksDataset(Dataset):
    def __init__(self, image_paths, image_size, augment):
        self.image_paths = image_paths
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]) if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.aug_transform is not None:
            img = self.aug_transform(img)
        return self.base_transform(img)


class ConvAE160(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.enc_block4 = nn.Sequential(
            nn.Conv2d(256, 640, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(640), nn.ReLU(inplace=True),
        )

        self.enc_flat_dim = 640 * 10 * 10
        self.bottleneck_dim = 2304

        self.fc_enc = nn.Linear(self.enc_flat_dim, self.bottleneck_dim)
        self.fc_latent = nn.Linear(self.bottleneck_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.bottleneck_dim)
        self.fc_dec2 = nn.Linear(self.bottleneck_dim, self.enc_flat_dim)

        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.dec_up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.dec_up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, return_skips=False):
        h1 = self.enc_block1(x)
        h2 = self.enc_block2(h1)
        h3 = self.enc_block3(h2)
        h4 = self.enc_block4(h3)

        h = F.relu(self.fc_enc(h4.view(h4.size(0), -1)))
        z = self.fc_latent(h)

        if return_skips:
            return z, (h1, h2, h3)
        return z

    def decode(self, z, skips=None):
        h = F.relu(self.fc_dec(z))
        h = F.relu(self.fc_dec2(h))
        h = h.view(h.size(0), 640, 10, 10)

        h = self.dec_up1(h)
        if skips is not None:
            h = h + skips[2]

        h = self.dec_up2(h)
        if skips is not None:
            h = h + skips[1]

        h = self.dec_up3(h)
        if skips is not None:
            h = h + skips[0]

        return self.dec_up4(h)

    def forward(self, x):
        z, skips = self.encode(x, return_skips=True)
        return self.decode(z, skips=skips), z


def ae_loss(recon_x, x):
    return 0.3 * F.mse_loss(recon_x, x) + 0.7 * F.l1_loss(recon_x, x)


def train_one_epoch(model, loader, optimizer, device, latent_noise):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        recon_main, z = model(batch)
        loss_main = ae_loss(recon_main, batch)

        z_noisy = z + latent_noise * torch.randn_like(z)
        recon_latent = model.decode(z_noisy, skips=None)
        loss_latent = ae_loss(recon_latent, batch)

        loss = 0.7 * loss_main + 0.3 * loss_latent
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        recon_main, z = model(batch)
        loss_main = ae_loss(recon_main, batch)
        recon_latent = model.decode(z, skips=None)
        loss_latent = ae_loss(recon_latent, batch)

        total_loss += (0.7 * loss_main + 0.3 * loss_latent).item()
        n += 1

    return total_loss / max(1, n)


@torch.no_grad()
def save_reconstructions(model, loader, device, out_path, num_images=8):
    model.eval()
    batch = next(iter(loader)).to(device)[:num_images]
    recon, _ = model(batch)
    grid = make_grid(torch.cat([batch.cpu(), recon.cpu()], dim=0), nrow=num_images, padding=2)
    save_image(grid, str(out_path))


@torch.no_grad()
def save_random_samples(model, dataset, device, out_path, num_samples, noise_z, noise_skip):
    model.eval()
    if len(dataset) == 0:
        return

    indices = np.random.choice(len(dataset), size=num_samples, replace=len(dataset) < num_samples)
    xs = torch.stack([dataset[int(i)] for i in indices]).to(device)

    z, skips = model.encode(xs, return_skips=True)
    h1, h2, h3 = skips

    if noise_z > 0:
        z = z + noise_z * torch.randn_like(z)
    if noise_skip > 0:
        h1 = h1 + noise_skip * torch.randn_like(h1)
        h2 = h2 + noise_skip * torch.randn_like(h2)
        h3 = h3 + noise_skip * torch.randn_like(h3)

    samples = model.decode(z, skips=(h1, h2, h3)).cpu()
    grid = make_grid(samples, nrow=int(num_samples ** 0.5) or 1, padding=2)
    save_image(grid, str(out_path))


def slerp(z1, z2, t):
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-0.999, 0.999)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if torch.any(sin_omega < 1e-6):
        return (1.0 - t) * z1 + t * z2

    t_tensor = torch.tensor(t, device=z1.device, dtype=z1.dtype).view(1, 1)
    factor1 = torch.sin((1.0 - t_tensor) * omega) / sin_omega
    factor2 = torch.sin(t_tensor * omega) / sin_omega
    return factor1 * z1 + factor2 * z2


@torch.no_grad()
def save_interpolation_gif(model, dataset, device, out_path, num_steps=24, frame_duration=0.18):
    if len(dataset) < 2:
        return

    model.eval()
    idx1, idx2 = random.sample(range(len(dataset)), 2)
    x1 = dataset[idx1].unsqueeze(0).to(device)
    x2 = dataset[idx2].unsqueeze(0).to(device)

    z1, skips1 = model.encode(x1, return_skips=True)
    z2, skips2 = model.encode(x2, return_skips=True)

    frames = []
    for step in range(num_steps):
        t = step / (num_steps - 1)
        z_t = slerp(z1, z2, t)
        skips_t = tuple((1.0 - t) * s1 + t * s2 for s1, s2 in zip(skips1, skips2))
        x_t = model.decode(z_t, skips=skips_t).cpu()[0]
        img = (x_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(img)

    pingpong = list(range(num_steps)) + list(range(num_steps - 2, 0, -1))
    with imageio.get_writer(str(out_path), mode="I", duration=frame_duration, loop=0) as writer:
        for idx in pingpong:
            writer.append_data(frames[idx])


def parse_args():
    parser = argparse.ArgumentParser(description="Convolutional AE for Minecraft block textures")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=176)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--latent-noise-train", type=float, default=0.15)
    parser.add_argument("--latent-noise-samples", type=float, default=0.25)
    parser.add_argument("--frame-duration", type=float, default=0.18)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    upscaled_dir = Path("upscaled")
    all_paths = upscale_raw(args.raw_dir, upscaled_dir, args.image_size)

    if not all_paths:
        raise RuntimeError("No .png files found for training")

    indices = list(range(len(all_paths)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * args.val_split))
    train_paths = [all_paths[i] for i in indices[val_size:]]
    val_paths = [all_paths[i] for i in indices[:val_size]]

    train_dataset = BlocksDataset(train_paths, args.image_size, augment=True)
    val_dataset = BlocksDataset(val_paths, args.image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = ConvAE160(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.latent_noise_train)
        val_loss = eval_one_epoch(model, val_loader, device)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"E{epoch:02d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            save_reconstructions(model, val_loader, device, args.out_dir / f"recon_{args.image_size}_epoch_{epoch:03d}.png")
            save_random_samples(
                model, val_dataset, device,
                args.out_dir / f"samples_{args.image_size}_epoch_{epoch:03d}.png",
                num_samples=36, noise_z=args.latent_noise_samples, noise_skip=0.03,
            )

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epochs + 1), train_loss_hist, label="train_loss")
    plt.plot(range(1, args.epochs + 1), val_loss_hist, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(args.out_dir / "losses.png")
    plt.close()

    save_interpolation_gif(model, val_dataset, device, args.out_dir / f"latent_interpolation_{args.image_size}.gif",
                           num_steps=24, frame_duration=args.frame_duration)


if __name__ == "__main__":
    main()
