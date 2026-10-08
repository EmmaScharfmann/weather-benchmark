"""The downscaling model.
The network predicts the correction to add to the interpolated coarse field,
and takes noise channels so that repeated passes give different plausible maps.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CHANNELS, TIME_CHANNELS


class _Block(nn.Module):
    """Two convolutions with group normalisation and a smooth activation."""

    def __init__(self, channels_in: int, channels_out: int):
        """
        :param channels_in: Input channels.
        :type channels_in: int
        :param channels_out: Output channels.
        :type channels_out: int
        """
        super().__init__()
        groups = min(8, channels_out)
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1), nn.GroupNorm(groups, channels_out), nn.SiLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1), nn.GroupNorm(groups, channels_out), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input of shape ``(batch, channels_in, height, width)``.
        :type x: torch.Tensor
        :returns: Output of shape ``(batch, channels_out, height, width)``.
        :rtype: torch.Tensor
        """
        return self.net(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections, zero-initialised output layer."""

    def __init__(self, channels_in: int, channels_out: int = 1, base: int = 32, depth: int = 3):
        """
        :param channels_in: Input channels.
        :type channels_in: int
        :param channels_out: Output channels.
        :type channels_out: int
        :param base: Channel count at full resolution.
        :type base: int
        :param depth: Number of downsampling steps.
        :type depth: int
        """
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.inc = _Block(channels_in, widths[0])
        self.down = nn.ModuleList([_Block(widths[i], widths[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.dec = nn.ModuleList([_Block(2 * widths[i], widths[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(widths[0], channels_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input of shape ``(batch, channels_in, height, width)``.
        :type x: torch.Tensor
        :returns: Output of shape ``(batch, channels_out, height, width)``.
        :rtype: torch.Tensor
        """
        skips = [self.inc(x)]
        for block in self.down:
            skips.append(block(F.avg_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], 1))
        return self.out(h)


@dataclass
class Config:
    """Model architecture settings."""

    loss: str = "crps"
    base: int = 32
    depth: int = 3
    noise_channels: int = 4

    @property
    def stochastic(self) -> bool:
        """Whether repeated passes give different maps.

        :returns: ``True`` for a CRPS-trained model with noise channels.
        :rtype: bool
        """
        return self.loss == "crps" and self.noise_channels > 0


class DownscalerNet(nn.Module):
    """A :class:`UNet` with noise channels, producing several maps per input."""

    def __init__(self, cfg: Config):
        """
        :param cfg: Architecture settings.
        :type cfg: Config
        """
        super().__init__()
        self.cfg = cfg
        n_noise = cfg.noise_channels if cfg.stochastic else 0
        self.unet = UNet(len(CHANNELS) + n_noise, 1, cfg.base, cfg.depth)

    def _noise(self, n: int, height: int, width: int, device) -> torch.Tensor:
        """Draw noise channels at two scales.

        Half the channels are drawn at full resolution and half at one eighth
        and upsampled, so members differ coherently over tens of kilometres
        rather than only pixel by pixel.

        :param n: Batch size.
        :type n: int
        :param height: Grid height.
        :type height: int
        :param width: Grid width.
        :type width: int
        :param device: Torch device.
        :type device: torch.device | str
        :returns: Noise of shape ``(n, noise_channels, height, width)``.
        :rtype: torch.Tensor
        """
        k = self.cfg.noise_channels
        fine = torch.randn(n, k // 2, height, width, device=device)
        coarse = torch.randn(n, k - k // 2, math.ceil(height / 8), math.ceil(width / 8), device=device)
        coarse = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
        return torch.cat([fine, coarse], 1)

    def forward(self, x: torch.Tensor, members: int = 1) -> torch.Tensor:
        """Predict normalised corrections.

        :param x: Input of shape ``(batch, channels, height, width)``.
        :type x: torch.Tensor
        :param members: Maps to produce per input; ignored by a deterministic model.
        :type members: int
        :returns: Corrections of shape ``(batch, members, height, width)``.
        :rtype: torch.Tensor
        """
        batch, _, height, width = x.shape
        m = members if self.cfg.stochastic else 1
        x = x.repeat_interleave(m, 0)
        if self.cfg.stochastic:
            x = torch.cat([x, self._noise(batch * m, height, width, x.device)], 1)
        multiple = 2 ** self.cfg.depth
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = self.unet(x)[..., :height, :width]
        return y.reshape(batch, m, height, width)


def fair_crps(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fair continuous ranked probability score, averaged over pixels.

    :param pred: Ensemble of shape ``(batch, members, height, width)``.
    :type pred: torch.Tensor
    :param target: Truth of shape ``(batch, height, width)``.
    :type target: torch.Tensor
    :returns: Scalar loss.
    :rtype: torch.Tensor
    :raises ValueError: If fewer than two members are given.
    """
    m = pred.shape[1]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    skill = (pred - target[:, None]).abs().mean(1)
    spread = (pred[:, :, None] - pred[:, None]).abs().sum((1, 2)) / (2 * m * (m - 1))
    return (skill - spread).mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the first member.

    :param pred: Prediction of shape ``(batch, members, height, width)``.
    :type pred: torch.Tensor
    :param target: Truth of shape ``(batch, height, width)``.
    :type target: torch.Tensor
    :returns: Scalar loss.
    :rtype: torch.Tensor
    """
    return ((pred[:, 0] - target) ** 2).mean()


def static_stack(static: dict, stats: dict) -> np.ndarray:
    """Normalise the static channels.

    :param static: Fields ``cerra_orography``, ``era5_orography``, ``land_sea_mask``, ``lat`` and ``lon``.
    :type static: dict
    :param stats: Normalisation statistics with ``mean`` and ``std`` per field.
    :type stats: dict
    :returns: Array of shape ``(5, height, width)`` in
        :data:`~weather_pipeline.constants.STATIC_CHANNELS` order.
    :rtype: numpy.ndarray
    """
    orography = static["cerra_orography"]
    height_diff = static["cerra_orography"] - static["era5_orography"]
    return np.stack([
        (orography - stats["cerra_orography"]["mean"]) / stats["cerra_orography"]["std"],
        (height_diff - stats["orography_diff"]["mean"]) / stats["orography_diff"]["std"],
        static["land_sea_mask"],
        (static["lat"] - stats["lat"]["mean"]) / stats["lat"]["std"],
        (static["lon"] - stats["lon"]["mean"]) / stats["lon"]["std"],
    ]).astype("f4")


def time_features(times) -> np.ndarray:
    """Encode the time of day and day of year.

    :param times: Valid times.
    :type times: numpy.ndarray | collections.abc.Sequence
    :returns: Array of shape ``(n_times, 4)`` holding the sine and cosine of each cycle.
    :rtype: numpy.ndarray
    """
    t = np.asarray(times, dtype="datetime64[ns]")
    hour = (t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")
    day_of_year = (t - t.astype("datetime64[Y]")) / np.timedelta64(1, "D")
    daily, yearly = 2 * np.pi * hour / 24, 2 * np.pi * day_of_year / 365.25
    return np.stack([np.sin(daily), np.cos(daily), np.sin(yearly), np.cos(yearly)], 1).astype("f4")


def make_inputs(coarse_k: np.ndarray, times, static_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Assemble the normalised model input.

    :param coarse_k: Coarse field on the fine grid [K], shape ``(time, height, width)``.
    :type coarse_k: numpy.ndarray
    :param times: Valid times, one per leading index.
    :type times: numpy.ndarray | collections.abc.Sequence
    :param static_norm: Normalised static channels from :func:`static_stack`.
    :type static_norm: numpy.ndarray
    :param stats: Normalisation statistics.
    :type stats: dict
    :returns: Array of shape ``(time, channels, height, width)``.
    :rtype: numpy.ndarray
    """
    n_times, height, width = coarse_k.shape
    coarse = ((coarse_k - stats["era5_t2m"]["mean"]) / stats["era5_t2m"]["std"]).astype("f4")[:, None]
    static = np.broadcast_to(static_norm[None], (n_times, *static_norm.shape))
    calendar = np.broadcast_to(
        time_features(times)[:, :, None, None], (n_times, len(TIME_CHANNELS), height, width)
    )
    return np.concatenate([coarse, static, calendar], 1)


def residual_norm(coarse_k: np.ndarray, fine_k: np.ndarray, stats: dict) -> np.ndarray:
    """Normalise the training target, which is the fine minus coarse difference.

    :param coarse_k: Coarse field on the fine grid [K].
    :type coarse_k: numpy.ndarray
    :param fine_k: Fine field [K].
    :type fine_k: numpy.ndarray
    :param stats: Normalisation statistics holding ``residual``.
    :type stats: dict
    :returns: Normalised residual, same shape as the inputs.
    :rtype: numpy.ndarray
    """
    residual = stats["residual"]
    return ((fine_k - coarse_k - residual["mean"]) / residual["std"]).astype("f4")


def to_kelvin(coarse_k: np.ndarray, pred_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Turn normalised corrections back into temperatures.

    :param coarse_k: Coarse field [K], shape ``(batch, height, width)``.
    :type coarse_k: numpy.ndarray
    :param pred_norm: Normalised corrections, shape ``(batch, members, height, width)``.
    :type pred_norm: numpy.ndarray
    :param stats: Normalisation statistics holding ``residual``.
    :type stats: dict
    :returns: Fine field [K], shape ``(batch, members, height, width)``.
    :rtype: numpy.ndarray
    """
    residual = stats["residual"]
    return coarse_k[:, None] + pred_norm * residual["std"] + residual["mean"]


class Downscaler:
    """A trained model together with everything inference needs."""

    def __init__(self, net: DownscalerNet, stats: dict, static: dict, device: str = "cpu"):
        """
        :param net: Trained network.
        :type net: DownscalerNet
        :param stats: Normalisation statistics.
        :type stats: dict
        :param static: Fine-grid static fields.
        :type static: dict
        :param device: Torch device.
        :type device: str
        """
        self.net = net.to(device).eval()
        self.stats = stats
        self.static = static
        self.device = device
        self.static_norm = static_stack(static, stats)
        self.lat, self.lon = static["lat"], static["lon"]

    def save(self, path, extra: dict | None = None) -> None:
        """Write the checkpoint.

        :param path: Destination file.
        :type path: str | pathlib.Path
        :param extra: Additional metadata to store alongside the weights.
        :type extra: dict | None
        :returns: ``None``
        :rtype: None
        """
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": asdict(self.net.cfg),
            "stats": self.stats,
            "static": {k: np.asarray(v) for k, v in self.static.items()},
            "channels": CHANNELS,
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "Downscaler":
        """Read a checkpoint.

        :param path: Checkpoint file.
        :type path: str | pathlib.Path
        :param device: Torch device; ``None`` selects CUDA when available.
        :type device: str | None
        :returns: Model ready for inference.
        :rtype: Downscaler
        :raises ValueError: If the checkpoint's channel order differs from this code's.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["channels"] != CHANNELS:
            raise ValueError(
                f"checkpoint channels {checkpoint['channels']} differ from this code's {CHANNELS}"
            )
        net = DownscalerNet(Config(**checkpoint["config"]))
        net.load_state_dict(checkpoint["state_dict"])
        return cls(net, checkpoint["stats"], checkpoint["static"], device)

    @torch.no_grad()
    def predict(self, coarse_k: np.ndarray, times, members: int = 4, batch: int = 8) -> np.ndarray:
        """Downscale a coarse field already on the fine grid.

        :param coarse_k: Coarse field [K], shape ``(time, height, width)``.
        :type coarse_k: numpy.ndarray
        :param times: Valid times, one per leading index.
        :type times: numpy.ndarray | collections.abc.Sequence
        :param members: Maps per input; a deterministic model always returns one.
        :type members: int
        :param batch: Time steps processed at once.
        :type batch: int
        :returns: Fine field [K], shape ``(time, members, height, width)``.
        :rtype: numpy.ndarray
        """
        out = []
        for i in range(0, len(coarse_k), batch):
            coarse = coarse_k[i:i + batch]
            x = torch.from_numpy(
                make_inputs(coarse, times[i:i + batch], self.static_norm, self.stats)
            ).to(self.device)
            pred = self.net(x, members).cpu().numpy()
            out.append(to_kelvin(coarse, pred, self.stats).astype("f4"))
        return np.concatenate(out, 0)