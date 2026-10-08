"""
The downscaler: coarse 2 m temperature (on the fine grid) + fine terrain ->
fine 2 m temperature, as an ENSEMBLE of plausible maps.

    model input (channels on the fine grid, all normalised)
        era5_t2m          coarse temperature, bilinearly interpolated   (varies in time)
        cerra_orography   fine terrain height
        orography_diff    fine minus coarse surface height  -> "how much higher is this pixel"
        land_sea_mask
        lat, lon          where we are (the model is regional)
        hour, day-of-year as sin/cos                             (varies in time)
        + noise channels  a fresh draw per member -> a different plausible map each time
    model output
        the CORRECTION to add to the coarse field (fine - coarse), in normalised units

Two losses (same network):
    "crps"  fair CRPS over M members per input: rewards being close to the truth AND
            having realistic spread, so detail is kept instead of averaged away
    "mse"   one deterministic map, the conditional mean: smooth by construction

Inference (step 6): Downscaler.load("model.pt").predict(coarse_K, times, members=4)
returns (time, member, y, x) in K. Everything needed (weights, normalisation,
fine-grid terrain, lat/lon) is inside the one checkpoint file.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATIC_CHANNELS = ["cerra_orography", "orography_diff", "land_sea_mask", "lat", "lon"]
TIME_CHANNELS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
CHANNELS = ["era5_t2m", *STATIC_CHANNELS, *TIME_CHANNELS]


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = min(8, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int = 1, base: int = 32, depth: int = 3):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.inc = _Block(cin, ch[0])
        self.down = nn.ModuleList([_Block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_Block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(ch[0], cout, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)     # start as "no correction"

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(F.avg_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], 1))
        return self.out(h)


@dataclass
class Config:
    loss: str = "crps"            # "crps" or "mse"
    base: int = 32
    depth: int = 3
    noise_channels: int = 4       # 0 for mse

    @property
    def stochastic(self) -> bool:
        return self.loss == "crps" and self.noise_channels > 0


class DownscalerNet(nn.Module):
    """UNet + noise. forward(x, members) -> (batch, members, H, W)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_noise = cfg.noise_channels if cfg.stochastic else 0
        self.unet = UNet(len(CHANNELS) + n_noise, 1, cfg.base, cfg.depth)

    def _noise(self, n, h, w, device):
        k = self.cfg.noise_channels
        fine = torch.randn(n, k // 2, h, w, device=device)
        # half the noise at 1/8 resolution: lets members differ coherently over ~50 km,
        # not only pixel by pixel
        coarse = torch.randn(n, k - k // 2, math.ceil(h / 8), math.ceil(w / 8), device=device)
        coarse = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
        return torch.cat([fine, coarse], 1)

    def forward(self, x: torch.Tensor, members: int = 1) -> torch.Tensor:
        b, _, h, w = x.shape
        m = members if self.cfg.stochastic else 1
        x = x.repeat_interleave(m, 0)
        if self.cfg.stochastic:
            x = torch.cat([x, self._noise(b * m, h, w, x.device)], 1)
        mult = 2 ** self.cfg.depth
        ph, pw = (-h) % mult, (-w) % mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        y = self.unet(x)[..., :h, :w]
        return y.reshape(b, m, h, w)


# ---------------------------------------------------------------------------
# losses (inputs in normalised units)
# ---------------------------------------------------------------------------
def fair_crps(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred (B, M, H, W), target (B, H, W). Fair (ensemble-size-unbiased) CRPS,
    averaged over pixels: mean|x_i - y| - sum_ij |x_i - x_j| / (2 M (M-1))."""
    m = pred.shape[1]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    skill = (pred - target[:, None]).abs().mean(1)
    spread = (pred[:, :, None] - pred[:, None]).abs().sum((1, 2)) / (2 * m * (m - 1))
    return (skill - spread).mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred[:, 0] - target) ** 2).mean()


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def static_stack(static: dict, stats: dict) -> np.ndarray:
    """(5, H, W) normalised static channels, in STATIC_CHANNELS order."""
    oro, dh = static["cerra_orography"], static["cerra_orography"] - static["era5_orography"]
    s = stats
    return np.stack([
        (oro - s["cerra_orography"]["mean"]) / s["cerra_orography"]["std"],
        (dh - s["orography_diff"]["mean"]) / s["orography_diff"]["std"],
        static["land_sea_mask"],
        (static["lat"] - s["lat"]["mean"]) / s["lat"]["std"],
        (static["lon"] - s["lon"]["mean"]) / s["lon"]["std"],
    ]).astype("f4")


def time_features(times) -> np.ndarray:
    """(T, 4): sin/cos of UTC hour and of day of year."""
    t = np.asarray(times, dtype="datetime64[ns]")
    hour = (t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")
    doy = (t - t.astype("datetime64[Y]")) / np.timedelta64(1, "D")
    a, b = 2 * np.pi * hour / 24, 2 * np.pi * doy / 365.25
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], 1).astype("f4")


def make_inputs(coarse_K: np.ndarray, times, static_norm: np.ndarray, stats: dict) -> np.ndarray:
    """coarse_K (T, H, W) -> (T, C, H, W) normalised model input."""
    t, h, w = coarse_K.shape
    e = ((coarse_K - stats["era5_t2m"]["mean"]) / stats["era5_t2m"]["std"]).astype("f4")[:, None]
    st = np.broadcast_to(static_norm[None], (t, *static_norm.shape))
    tf = np.broadcast_to(time_features(times)[:, :, None, None], (t, len(TIME_CHANNELS), h, w))
    return np.concatenate([e, st, tf], 1)


def residual_norm(coarse_K, fine_K, stats):
    r = stats["residual"]
    return ((fine_K - coarse_K - r["mean"]) / r["std"]).astype("f4")


def to_kelvin(coarse_K, pred_norm, stats):
    """coarse (B, H, W) + normalised correction (B, M, H, W) -> fine (B, M, H, W) in K."""
    r = stats["residual"]
    return coarse_K[:, None] + pred_norm * r["std"] + r["mean"]


# ---------------------------------------------------------------------------
# save / load / predict
# ---------------------------------------------------------------------------
class Downscaler:
    """A trained model, ready to apply to any coarse field already on the fine grid."""

    def __init__(self, net: DownscalerNet, stats: dict, static: dict, device="cpu"):
        self.net, self.stats, self.static, self.device = net.to(device).eval(), stats, static, device
        self.static_norm = static_stack(static, stats)
        self.lat, self.lon = static["lat"], static["lon"]

    def save(self, path, extra: dict | None = None):
        torch.save({"state_dict": self.net.state_dict(), "config": asdict(self.net.cfg), "stats": self.stats,
                    "static": {k: np.asarray(v) for k, v in self.static.items()}, "channels": CHANNELS,
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["channels"] != CHANNELS:
            raise ValueError(f"checkpoint channels {ck['channels']} differ from this code's {CHANNELS}")
        net = DownscalerNet(Config(**ck["config"]))
        net.load_state_dict(ck["state_dict"])
        return cls(net, ck["stats"], ck["static"], device)

    @torch.no_grad()
    def predict(self, coarse_K: np.ndarray, times, members: int = 4, batch: int = 8) -> np.ndarray:
        """coarse_K (T, H, W) in K, on the fine grid -> (T, members, H, W) in K.
        A deterministic (mse) model returns members=1 whatever is asked."""
        out = []
        for i in range(0, len(coarse_K), batch):
            c = coarse_K[i:i + batch]
            x = torch.from_numpy(make_inputs(c, times[i:i + batch], self.static_norm, self.stats)).to(self.device)
            p = self.net(x, members).cpu().numpy()
            out.append(to_kelvin(c, p, self.stats).astype("f4"))
        return np.concatenate(out, 0)