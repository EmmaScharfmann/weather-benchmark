"""Tests for the downscaling model, its loss and its checkpoint."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is installed with requirements/downscaler.txt")

from downscale import downscaler as dsc  # noqa: E402
from downscale.constants import CHANNELS  # noqa: E402


def test_fair_crps_matches_the_explicit_formula():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(3, 5, 6, 7)).astype("f4")
    target = rng.normal(size=(3, 6, 7)).astype("f4")
    m = pred.shape[1]
    skill = np.abs(pred - target[:, None]).mean(1)
    spread = np.abs(pred[:, :, None] - pred[:, None]).sum((1, 2)) / (2 * m * (m - 1))
    value = float(dsc.fair_crps(torch.from_numpy(pred), torch.from_numpy(target)))
    assert value == pytest.approx((skill - spread).mean(), rel=1e-5)


def test_fair_crps_of_a_perfect_gaussian_ensemble():
    torch.manual_seed(0)
    ensemble, truth = torch.randn(20000, 16, 1, 1), torch.randn(20000, 1, 1)
    assert float(dsc.fair_crps(ensemble, truth)) == pytest.approx(1 / np.sqrt(np.pi), abs=0.01)


def test_fair_crps_needs_at_least_two_members():
    with pytest.raises(ValueError, match="at least 2 members"):
        dsc.fair_crps(torch.zeros(2, 1, 3, 3), torch.zeros(2, 3, 3))


def test_time_features_encode_the_daily_cycle():
    times = np.array(["2020-07-01T00", "2020-07-01T12"], "datetime64[ns]")
    features = dsc.time_features(times)
    assert np.allclose(features[0, :2], [0, 1], atol=1e-6)
    assert np.allclose(features[1, :2], [0, -1], atol=1e-6)


def test_inputs_have_one_channel_per_name(static_fields, stats):
    height, width = static_fields["lat"].shape
    coarse = np.full((3, height, width), 290.0, "f4")
    times = np.array(["2020-07-01T00", "2020-07-01T12", "2020-12-31T18"], "datetime64[ns]")
    inputs = dsc.make_inputs(coarse, times, dsc.static_stack(static_fields, stats), stats)
    assert inputs.shape == (3, len(CHANNELS), height, width)


@pytest.mark.parametrize("loss", ["crps", "mse"])
def test_network_shapes_and_determinism(loss, static_fields, stats):
    height, width = static_fields["lat"].shape
    coarse = np.full((3, height, width), 290.0, "f4")
    times = np.array(["2020-07-01T00", "2020-07-01T12", "2020-12-31T18"], "datetime64[ns]")
    inputs = torch.from_numpy(dsc.make_inputs(coarse, times, dsc.static_stack(static_fields, stats), stats))
    net = dsc.DownscalerNet(dsc.Config(loss=loss, base=8, depth=3, noise_channels=4 if loss == "crps" else 0))
    out = net(inputs, members=4)
    assert out.shape == ((3, 4, height, width) if loss == "crps" else (3, 1, height, width))
    assert torch.allclose(out, torch.zeros_like(out))
    with torch.no_grad():
        for parameter in net.unet.out.parameters():
            parameter.normal_(0, 0.1)
    first, second = net(inputs, 4), net(inputs, 4)
    assert (not torch.allclose(first, second)) if loss == "crps" else torch.allclose(first, second)


def test_odd_grid_sizes_are_padded_and_cropped_back(stats):
    height, width = 37, 45
    static = {
        "cerra_orography": np.zeros((height, width), "f4"),
        "era5_orography": np.zeros((height, width), "f4"),
        "land_sea_mask": np.ones((height, width), "f4"),
        "lat": np.linspace(41, 51, height)[:, None].repeat(width, 1).astype("f4"),
        "lon": np.linspace(-5, 10, width)[None].repeat(height, 0).astype("f4"),
    }
    local_stats = dict(stats)
    local_stats["cerra_orography"] = {"mean": 0.0, "std": 1.0}
    local_stats["orography_diff"] = {"mean": 0.0, "std": 1.0}
    local_stats["lat"] = {"mean": 46.0, "std": 3.0}
    local_stats["lon"] = {"mean": 2.0, "std": 4.0}
    coarse = np.full((2, height, width), 290.0, "f4")
    times = np.array(["2020-07-01T00", "2020-07-01T12"], "datetime64[ns]")
    inputs = torch.from_numpy(dsc.make_inputs(coarse, times, dsc.static_stack(static, local_stats), local_stats))
    out = dsc.DownscalerNet(dsc.Config(base=8)).forward(inputs, members=2)
    assert out.shape == (2, 2, height, width)


@pytest.mark.parametrize("loss", ["crps", "mse"])
def test_checkpoint_round_trip_and_prediction(loss, tmp_path, static_fields, stats):
    height, width = static_fields["lat"].shape
    net = dsc.DownscalerNet(dsc.Config(loss=loss, base=8, noise_channels=4 if loss == "crps" else 0))
    model = dsc.Downscaler(net, stats, static_fields)
    model.save(tmp_path / "model.pt")
    restored = dsc.Downscaler.load(tmp_path / "model.pt", device="cpu")
    coarse = np.full((3, height, width), 290.0, "f4")
    times = np.array(["2020-07-01T00", "2020-07-01T12", "2020-12-31T18"], "datetime64[ns]")
    prediction = restored.predict(coarse, times, members=3, batch=2)
    expected_members = 3 if loss == "crps" else 1
    assert prediction.shape == (3, expected_members, height, width)
    assert np.isfinite(prediction).all()


def test_loading_a_checkpoint_with_other_channels_raises(tmp_path, static_fields, stats):
    model = dsc.Downscaler(dsc.DownscalerNet(dsc.Config(base=8)), stats, static_fields)
    model.save(tmp_path / "model.pt")
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
    checkpoint["channels"] = ["something_else"]
    torch.save(checkpoint, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="channels"):
        dsc.Downscaler.load(tmp_path / "model.pt", device="cpu")


def test_residual_and_kelvin_conversions_are_inverse(stats):
    coarse = np.full((2, 4, 5), 290.0, "f4")
    fine = coarse + 1.5
    normalised = dsc.residual_norm(coarse, fine, stats)
    recovered = dsc.to_kelvin(coarse, normalised[:, None], stats)
    assert np.allclose(recovered[:, 0], fine, atol=1e-4)