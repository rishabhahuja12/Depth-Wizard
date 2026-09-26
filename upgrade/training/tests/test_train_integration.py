"""
CPU end-to-end integration test of the real training path.

Creates a tiny FAKE local tile cache (small h5 files laid out exactly where
hf_hub_download would place them), then patches list_repo_files and
hf_hub_download so the dataset reads from that cache instead of hitting the
network. Runs the actual train_metric.train() on CPU with the small model,
exercising the dataset, loss, loop, held-out eval, checkpoint SAVE and
checkpoint RESUME. No GPU, no network.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_train_integration.py
"""
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(TRAIN_DIR.parent / "evaluation"))

import train_metric as tm  # noqa: E402

SMALL = tm.SMALL_MODEL_ID
TILE = 256


def _write_fake_tile(cache: Path, split: str, name: str) -> str:
    """Write a fake h5 pair in the layout hf_hub_download uses (local_dir/rel_path)."""
    import h5py
    rgb_rel = f"images/{split}/{name}_RGB.h5"
    agl_rel = f"heights/{split}/{name}_AGL.h5"
    rgb_p = cache / rgb_rel
    agl_p = cache / agl_rel
    rgb_p.parent.mkdir(parents=True, exist_ok=True)
    agl_p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    rgb = (rng.random((TILE, TILE, 3)) * 255).astype(np.uint8)
    agl = (rng.random((TILE, TILE)).astype(np.float32) * 30.0)
    with h5py.File(rgb_p, "w") as f:
        f.create_dataset("image", data=rgb)
    with h5py.File(agl_p, "w") as f:
        f.create_dataset("image", data=agl)
    return rgb_rel


def _build_cache(cache: Path):
    """Write two train tiles and one val tile into cache."""
    _write_fake_tile(cache, "train", "DC_0")
    _write_fake_tile(cache, "train", "DC_1")
    _write_fake_tile(cache, "val", "JAX_0")


def _fake_list_repo_files(repo, repo_type=None):
    """Deterministic file listing that matches _build_cache output."""
    return [
        "images/train/DC_0_RGB.h5",
        "images/train/DC_1_RGB.h5",
        "heights/train/DC_0_AGL.h5",
        "heights/train/DC_1_AGL.h5",
        "images/val/JAX_0_RGB.h5",
        "heights/val/JAX_0_AGL.h5",
    ]


def _make_fake_hf_download(cache: Path):
    """Return a mock hf_hub_download that resolves paths from local cache."""
    import huggingface_hub
    orig_download = getattr(huggingface_hub, "_orig_download", huggingface_hub.hf_hub_download)
    def _download(repo, filename, *args, **kwargs):
        local_dir = kwargs.get("local_dir")
        p = Path(local_dir or cache) / filename
        if p.exists():
            return str(p)
        return orig_download(repo, filename, *args, **kwargs)
    return _download


def _args(cache: Path, ckpt: Path, epochs: int, resume: bool, oc_root=None):
    return Namespace(
        model=tm.LARGE_MODEL_ID, epochs=epochs, batch=1, grad_accum=1, crop=128, tile_size=TILE,
        warmup=1, enc_lr=5e-6, head_lr=5e-5, w_silog=1.0, w_l1=1.0, w_grad=0.0, w_lt=0.0,
        workers=0, val_tiles=1, max_vram_frac=0.0, throttle_sleep=0.0,
        augment=False, lora=False, dora=False,
        cache_dir=str(cache), gamus_root=str(cache), offline=True, ckpt_dir=str(ckpt), resume=resume,
        blend=bool(oc_root), oc_root=oc_root, gbh_root=None,
        aux_fraction=0.5, aux_gsd=0.5,
    )


def _write_fake_open_canopy(root: Path, n=2, size=48, h_val=15.0):
    """Fake Open-Canopy layout: images/*.tif (RGB) + canopy_height/*.tif (meters)."""
    import rasterio
    from rasterio.transform import from_origin
    for i in range(n):
        rgb = (np.ones((3, size, size)) * 120).astype(np.uint8)
        h = (np.ones((size, size)) * h_val).astype(np.float32)
        for sub, arr, count in (("images", rgb, 3), ("canopy_height", h[None], 1)):
            p = root / sub / f"oc_{i}.tif"
            p.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=count,
                               dtype=arr.dtype, transform=from_origin(0, 0, 1, 1)) as ds:
                for b in range(count):
                    ds.write(arr[b], b + 1)


def test_train_runs_and_writes_checkpoints_on_cpu():
    with tempfile.TemporaryDirectory() as d:
        cache, ckpt = Path(d) / "cache", Path(d) / "ckpt"
        _build_cache(cache)
        with (patch("huggingface_hub.list_repo_files", side_effect=_fake_list_repo_files),
              patch("huggingface_hub.hf_hub_download", side_effect=_make_fake_hf_download(cache))):
            rc = tm.train(_args(cache, ckpt, epochs=1, resume=False))
        assert rc == 0, "train() returned nonzero"
        assert (ckpt / "metric_last.pth").exists(), "no metric_last.pth"
        assert (ckpt / "metric_best.pth").exists(), "no metric_best.pth"


def test_resume_continues_from_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        cache, ckpt = Path(d) / "cache", Path(d) / "ckpt"
        _build_cache(cache)
        with (patch("huggingface_hub.list_repo_files", side_effect=_fake_list_repo_files),
              patch("huggingface_hub.hf_hub_download", side_effect=_make_fake_hf_download(cache))):
            tm.train(_args(cache, ckpt, epochs=1, resume=False))
            import torch
            done1 = torch.load(ckpt / "metric_last.pth", map_location="cpu", weights_only=False)["epoch"]
            assert done1 == 1
            tm.train(_args(cache, ckpt, epochs=2, resume=True))
            done2 = torch.load(ckpt / "metric_last.pth", map_location="cpu", weights_only=False)["epoch"]
            assert done2 == 2, f"resume didn't advance: {done2}"


def test_blend_runs_gamus_plus_open_canopy_on_cpu():
    with tempfile.TemporaryDirectory() as d:
        cache, ckpt, oc = Path(d) / "cache", Path(d) / "ckpt", Path(d) / "oc"
        _build_cache(cache)
        _write_fake_open_canopy(oc, n=2, h_val=15.0)
        with (patch("huggingface_hub.list_repo_files", side_effect=_fake_list_repo_files),
              patch("huggingface_hub.hf_hub_download", side_effect=_make_fake_hf_download(cache))):
            rc = tm.train(_args(cache, ckpt, epochs=1, resume=False, oc_root=str(oc)))
        assert rc == 0, "blended train() returned nonzero"
        assert (ckpt / "metric_best.pth").exists(), "no checkpoint from blended run"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok    {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
