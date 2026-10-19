import os, sys, glob, random, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

torch.set_num_threads(os.cpu_count() or 4)

PRESETS = {"quick": 2000, "normal": 10000, "long": 30000}
MODEL_FILE = "broken_model.pt"
PARTIAL_FILE = "broken_model.pt.partial"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.ReLU())

def up(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                         nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.ReLU())

class AE(nn.Module):
    def __init__(self, width=16, lat_ch=4):
        super().__init__()
        assert width % 8 == 0, "width должен делиться на 8"
        w = width
        self.enc = nn.Sequential(down(3, w), down(w, 2 * w), down(2 * w, 4 * w),
                                 down(4 * w, 4 * w), nn.Conv2d(4 * w, lat_ch, 3, 1, 1))
        self.dec = nn.Sequential(nn.Conv2d(lat_ch, 4 * w, 3, 1, 1), nn.ReLU(),
                                 up(4 * w, 4 * w), up(4 * w, 2 * w), up(2 * w, w), up(w, w),
                                 nn.Conv2d(w, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))

def loss_fn(a, b, kind):
    if kind == "mse":
        return F.mse_loss(a, b)
    l = F.l1_loss(a, b)
    ax, ay = a[:, :, :, 1:] - a[:, :, :, :-1], a[:, :, 1:, :] - a[:, :, :-1, :]
    bx, by = b[:, :, :, 1:] - b[:, :, :, :-1], b[:, :, 1:, :] - b[:, :, :-1, :]
    return l + 0.5 * (F.l1_loss(ax, bx) + F.l1_loss(ay, by))

def list_images(folder):
    files = glob.glob(os.path.join(folder, "**", "*"), recursive=True)
    return [f for f in files if os.path.splitext(f)[1].lower() in EXTS]

def load_data(args):
    if args.data:
        files = list_images(args.data)
        if not files:
            sys.exit(f"В {args.data} нет картинок")
        if len(files) > args.limit:
            files = random.sample(files, args.limit)
        print(f"Загружаю {len(files)} картинок из {args.data}...")
        arr = [np.asarray(Image.open(f).convert("RGB").resize((96, 96), Image.LANCZOS))
               for f in files]
        return np.stack(arr).transpose(0, 3, 1, 2)
    from torchvision.datasets import STL10
    print("Готовлю STL-10 (при первом запуске скачается ~2.6 ГБ)...")
    ds = STL10(root=args.cache, split="unlabeled", download=True)
    imgs = ds.data
    if len(imgs) > args.limit:
        imgs = imgs[np.random.choice(len(imgs), args.limit, replace=False)]
    return imgs

def get_batch(imgs, n, crop):
    x = torch.from_numpy(imgs[np.random.randint(0, len(imgs), n)]).float() / 255
    if crop < x.shape[-1]:
        i, j = random.randint(0, x.shape[-1] - crop), random.randint(0, x.shape[-1] - crop)
        x = x[:, :, i:i + crop, j:j + crop]
    if random.random() < 0.5:
        x = x.flip(3)
    return x

def train(args):
    assert args.crop % 16 == 0 and args.crop <= 96, "--crop кратен 16 и <= 96"
    steps = args.steps or PRESETS[args.preset]
    imgs = load_data(args)
    model = AE(args.width, args.lat_ch)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Модель: width={args.width}, lat_ch={args.lat_ch}, параметров: {n_par:,}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    start = 0
    if os.path.exists(PARTIAL_FILE):
        ck = torch.load(PARTIAL_FILE)
        if ck["cfg"] == {"width": args.width, "lat_ch": args.lat_ch}:
            model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); start = ck["step"]
            print(f"Продолжаю с шага {start}")
    os.makedirs("out", exist_ok=True)
    cfg = {"width": args.width, "lat_ch": args.lat_ch}
    t0 = time.time()
    for step in range(start, steps):
        x = get_batch(imgs, args.batch, args.crop)
        loss = loss_fn(model(x), x, args.loss)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{steps}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
        if (step + 1) % 1000 == 0:
            torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, "cfg": cfg},
                       PARTIAL_FILE)
    torch.save({"state": model.state_dict(), "cfg": cfg}, MODEL_FILE)
    if os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)
    with torch.no_grad():
        x = get_batch(imgs, 8, args.crop)
        save_image(torch.cat([x, model(x)]), "out/train_check.png", nrow=8)
    print(f"Готово: {MODEL_FILE}. Проверь out/train_check.png (верх: оригинал, низ: сеть)")

def fit_size(w, h, long_side):
    s = long_side / max(w, h)
    return max(16, round(w * s / 16) * 16), max(16, round(h * s / 16) * 16)

def apply(args):
    ck = torch.load(args.model)
    model = AE(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    os.makedirs("out", exist_ok=True)
    files = []
    for p in args.paths:
        files += list_images(p) if os.path.isdir(p) else [p]
    for path in files:
        im = Image.open(path).convert("RGB")
        nw, nh = fit_size(*im.size, args.long_side)
        small = im.resize((nw, nh), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad():
            y = x
            for _ in range(args.passes):
                y = model(y)
        arr = (y[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        broken = Image.fromarray(arr).resize((nw * args.scale, nh * args.scale), Image.BICUBIC)
        orig = small.resize(broken.size, Image.BICUBIC)
        name = os.path.splitext(os.path.basename(path))[0]
        broken.save(f"out/broken_{name}.png")
        both = Image.new("RGB", (broken.width * 2, broken.height))
        both.paste(orig, (0, 0)); both.paste(broken, (broken.width, 0))
        both.save(f"out/compare_{name}.png")
        print(f"{path} -> out/broken_{name}.png")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--preset", choices=list(PRESETS), default="normal")
    t.add_argument("--steps", type=int)
    t.add_argument("--data", help="своя папка картинок вместо STL-10")
    t.add_argument("--cache", default="./data", help="куда скачать STL-10")
    t.add_argument("--limit", type=int, default=30000, help="сколько картинок брать в память")
    t.add_argument("--width", type=int, default=16)
    t.add_argument("--lat-ch", type=int, default=4)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--crop", type=int, default=64)
    t.add_argument("--loss", choices=["l1edge", "mse"], default="l1edge")
    a = sub.add_parser("apply")
    a.add_argument("paths", nargs="+")
    a.add_argument("--model", default=MODEL_FILE)
    a.add_argument("--long-side", type=int, default=128, help="размер длинной стороны для сети")
    a.add_argument("--scale", type=int, default=4, help="во сколько раз увеличить результат")
    a.add_argument("--passes", type=int, default=1, help="сколько раз прогнать через сеть")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else apply(args)

if __name__ == "__main__":
    main()
