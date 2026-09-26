"""
=============================================================================
DEPTH-WIZARD AUXILIARY DATASET DOWNLOADER
=============================================================================

CURRENTLY ENABLED:
    1. Open-Canopy 2023
       Hugging Face: AI4Forest/Open-Canopy
       Local:        ./Open-Canopy

    2. DFC2023 Track 2
       IEEE DataPort
       Local:        ./DFC2023

CURRENTLY DISABLED:
    M4Heights

IMPORTANT DESIGN GOALS:
    - Never intentionally delete downloaded dataset files.
    - Resume Open-Canopy using the existing local directory.
    - Retry transient Hugging Face/network errors.
    - Use one concurrent download worker by default.
    - Keep a persistent status file.
    - Verify the expected Open-Canopy 2023 structure.
    - Do not falsely report a partial download as complete.
    - DFC2023 ZIP is preserved after extraction.
    - Safe to stop with Ctrl+C and rerun.
    - M4Heights is NOT touched.

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Ensure live unbuffered console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

# ============================================================================
# PATHS
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent

OC_DIR = ROOT_DIR / "Open-Canopy"
DFC_DIR = ROOT_DIR / "DFC2023"

STATUS_FILE = ROOT_DIR / "aux_dataset_status.json"

# ============================================================================
# HUGGING FACE
# ============================================================================

OC_REPO = "AI4Forest/Open-Canopy"

# ============================================================================
# DFC2023
# ============================================================================

DFC_URL = (
    "https://ieee-dataport.org/competitions/"
    "2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification"
)

# ============================================================================
# EXPECTED OPEN-CANOPY 2023
# ============================================================================

EXPECTED_OC_SPOT_COUNT = 18
EXPECTED_OC_LIDAR_COUNT = 18

# Approximate expected downloaded size.
# This is used as a warning/diagnostic, not as the sole completion criterion.
EXPECTED_OC_SIZE_GB = 75.35

# ============================================================================
# GENERAL UTILITIES
# ============================================================================


def bytes_to_gb(value: int | float) -> float:
    return value / (1024 ** 3)


def directory_size(path: Path) -> int:
    """Calculate total regular-file size under a directory."""

    if not path.exists():
        return 0

    total = 0

    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except (FileNotFoundError, PermissionError):
            pass

    return total


def count_files(path: Path) -> int:
    """Count regular files under a directory."""

    if not path.exists():
        return 0

    count = 0

    for p in path.rglob("*"):
        try:
            if p.is_file():
                count += 1
        except (FileNotFoundError, PermissionError):
            pass

    return count


def disk_space(path: Path) -> tuple[float, float, float]:
    """
    Return total, used, free space in GB.
    """

    total, used, free = shutil.disk_usage(str(path))

    return (
        bytes_to_gb(total),
        bytes_to_gb(used),
        bytes_to_gb(free),
    )


def format_bytes(value: int) -> str:
    """Human-readable byte size."""

    gb = bytes_to_gb(value)

    if gb >= 1:
        return f"{gb:.2f} GB"

    mb = value / (1024 ** 2)

    if mb >= 1:
        return f"{mb:.2f} MB"

    kb = value / 1024

    return f"{kb:.2f} KB"


# ============================================================================
# STATUS FILE
# ============================================================================


def load_status() -> dict:
    if not STATUS_FILE.exists():
        return {}

    try:
        return json.loads(
            STATUS_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def save_status(status: dict) -> None:
    """
    Atomically-ish replace status file.

    A temporary file is used so that an interrupted write is less likely
    to destroy the previous status file.
    """

    tmp = STATUS_FILE.with_suffix(".tmp")

    try:
        tmp.write_text(
            json.dumps(
                status,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        os.replace(tmp, STATUS_FILE)

    except Exception as e:
        print(
            f"[WARNING] Could not update status file: {e}"
        )

        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def update_status(dataset: str, **values) -> None:
    status = load_status()

    entry = status.get(dataset, {})

    entry.update(values)

    entry["updated_at"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    status[dataset] = entry

    save_status(status)


# ============================================================================
# DISK REPORT
# ============================================================================


def print_space_report() -> None:
    total, used, free = disk_space(ROOT_DIR)

    print("\n" + "=" * 72)
    print("DISK SPACE INSPECTION")
    print("=" * 72)

    print(f"Drive:             {Path(ROOT_DIR).drive}")
    print(f"Total capacity:    {total:.2f} GB")
    print(f"Used:              {used:.2f} GB")
    print(f"Available free:    {free:.2f} GB")

    print("-" * 72)

    oc_size = directory_size(OC_DIR)
    dfc_size = directory_size(DFC_DIR)

    print("CURRENT LOCAL DATA:")
    print(
        f"  Open-Canopy:     {bytes_to_gb(oc_size):.2f} GB"
    )
    print(
        f"  DFC2023:         {bytes_to_gb(dfc_size):.2f} GB"
    )

    print("-" * 72)

    print(
        f"Expected Open-Canopy 2023: "
        f"~{EXPECTED_OC_SIZE_GB:.2f} GB"
    )

    print(
        "\nM4Heights: DISABLED / NOT TOUCHED"
    )

    print("=" * 72)


# ============================================================================
# OPEN-CANOPY INSPECTION
# ============================================================================


def get_oc_2023_dirs() -> tuple[Path, Path]:
    spot_dir = (
        OC_DIR
        / "canopy_height"
        / "2023"
        / "spot"
    )

    lidar_dir = (
        OC_DIR
        / "canopy_height"
        / "2023"
        / "lidar"
    )

    return spot_dir, lidar_dir


def get_oc_tifs() -> tuple[list[Path], list[Path]]:
    spot_dir, lidar_dir = get_oc_2023_dirs()

    spot_files = (
        sorted(spot_dir.glob("*.tif"))
        if spot_dir.exists()
        else []
    )

    lidar_files = (
        sorted(lidar_dir.glob("*.tif"))
        if lidar_dir.exists()
        else []
    )

    return spot_files, lidar_files


def inspect_open_canopy() -> dict:
    """
    Inspect current Open-Canopy 2023 state.

    Returns a dictionary suitable for status reporting.
    """

    print("\n" + "=" * 72)
    print("OPEN-CANOPY 2023 LOCAL STATE")
    print("=" * 72)

    print(f"Directory: {OC_DIR}")

    if not OC_DIR.exists():
        print("Directory does not exist yet.")
        return {
            "exists": False,
            "complete": False,
            "spot_count": 0,
            "lidar_count": 0,
            "size_bytes": 0,
        }

    spot_files, lidar_files = get_oc_tifs()

    total_size = directory_size(OC_DIR)

    geometries = (
        OC_DIR
        / "canopy_height"
        / "geometries.geojson"
    )

    print(
        f"Local size:        "
        f"{bytes_to_gb(total_size):.2f} GB"
    )

    print(
        f"SPOT TIFFs:        {len(spot_files)}"
        f" / expected {EXPECTED_OC_SPOT_COUNT}"
    )

    print(
        f"LiDAR TIFFs:       {len(lidar_files)}"
        f" / expected {EXPECTED_OC_LIDAR_COUNT}"
    )

    print(
        f"geometries.geojson: "
        f"{'YES' if geometries.exists() else 'NO'}"
    )

    print("\nSPOT files:")

    for p in spot_files:
        try:
            size = format_bytes(p.stat().st_size)
        except Exception:
            size = "unknown"

        print(f"  {p.name}    {size}")

    print("\nLiDAR files:")

    for p in lidar_files:
        try:
            size = format_bytes(p.stat().st_size)
        except Exception:
            size = "unknown"

        print(f"  {p.name}    {size}")

    complete = (
        len(spot_files) >= EXPECTED_OC_SPOT_COUNT
        and len(lidar_files) >= EXPECTED_OC_LIDAR_COUNT
        and geometries.exists()
    )

    print("\nState:")

    if complete:
        print("  [POSSIBLY COMPLETE]")
    else:
        print("  [INCOMPLETE / DOWNLOAD NEEDED]")

    print("=" * 72)

    return {
        "exists": True,
        "complete": complete,
        "spot_count": len(spot_files),
        "lidar_count": len(lidar_files),
        "size_bytes": total_size,
        "geometries": geometries.exists(),
    }


# ============================================================================
# OPEN-CANOPY ORGANIZATION
# ============================================================================


def prepare_open_canopy_layout() -> None:
    """
    Create convenient paired links under:

        Open-Canopy/images/
        Open-Canopy/canopy_height/

    IMPORTANT:
        Hard links are preferred.
        Symlinks are second.
        Copying is last resort.

    Copying huge TIFFs can consume a large amount of additional disk space.
    """

    print("\n[Open-Canopy] Preparing paired layout...")

    images_dir = OC_DIR / "images"
    height_dir = OC_DIR / "canopy_height"

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    height_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spot_files = sorted(
        OC_DIR.rglob(
            "compressed_pansharpened_*.tif"
        )
    )

    lidar_files = sorted(
        OC_DIR.rglob(
            "compressed_lidar_*.tif"
        )
    )

    lidar_map = {
        p.name.replace(
            "compressed_lidar_",
            "",
        ): p
        for p in lidar_files
    }

    paired = 0

    for spot in spot_files:
        key = spot.name.replace(
            "compressed_pansharpened_",
            "",
        )

        lidar = lidar_map.get(key)

        if lidar is None:
            continue

        target_rgb = images_dir / (
            f"tile_{key}"
        )

        target_height = height_dir / (
            f"tile_{key}"
        )

        # --------------------------------------------------------------
        # RGB
        # --------------------------------------------------------------

        if not target_rgb.exists():
            if not create_link_or_copy(
                spot,
                target_rgb,
            ):
                print(
                    f"[WARNING] Could not create RGB link: "
                    f"{target_rgb}"
                )

        # --------------------------------------------------------------
        # Height
        # --------------------------------------------------------------

        if not target_height.exists():
            if not create_link_or_copy(
                lidar,
                target_height,
            ):
                print(
                    f"[WARNING] Could not create height link: "
                    f"{target_height}"
                )

        if (
            target_rgb.exists()
            and target_height.exists()
        ):
            paired += 1

    print(
        f"[Open-Canopy] Paired layout: "
        f"{paired} tile pair(s)."
    )


def create_link_or_copy(
    source: Path,
    target: Path,
) -> bool:
    """
    Try hard link -> symlink -> copy.

    Returns True on success.

    NOTE:
        Copying is deliberately last because it duplicates potentially
        huge TIFF files.
    """

    try:
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Same NTFS filesystem: hard link should normally work.
        os.link(
            str(source),
            str(target),
        )

        return True

    except Exception:
        pass

    try:
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.symlink(
            str(source),
            str(target),
        )

        return True

    except Exception:
        pass

    print(
        f"[WARNING] Hard/symbolic link unavailable."
    )

    print(
        f"[WARNING] FALLBACK COPY: {source.name}"
    )

    try:
        shutil.copy2(
            str(source),
            str(target),
        )

        return True

    except Exception as e:
        print(
            f"[ERROR] Copy failed: {e}"
        )

        return False


# ============================================================================
# OPEN-CANOPY DOWNLOAD
# ============================================================================


def open_canopy_patterns() -> list[str]:
    return [
        "canopy_height/2023/spot/*.tif",
        "canopy_height/2023/lidar/*.tif",
        "canopy_height/geometries.geojson",
    ]


def download_open_canopy_file(
    rfilename: str,
    expected_size: int | None = None,
    max_retries: int = 10,
    token: str | None = None,
) -> bool:
    """Download a single file into OC_DIR with retry logic and exact size verification."""
    from huggingface_hub import hf_hub_download
    dest_path = OC_DIR / Path(rfilename)
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size is None or actual_size == expected_size:
            print(f"    [OK / EXISTS] {dest_path.name} ({format_bytes(actual_size)})")
            return True
        else:
            print(f"    [INCOMPLETE] {dest_path.name} ({format_bytes(actual_size)} / expected {format_bytes(expected_size)}). Resuming...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=OC_REPO,
                filename=rfilename,
                repo_type="dataset",
                local_dir=str(OC_DIR),
                token=token,
            )
            if dest_path.exists():
                actual_size = dest_path.stat().st_size
                if expected_size is None or actual_size == expected_size:
                    print(f"    [COMPLETED] {dest_path.name} ({format_bytes(actual_size)})")
                    return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == max_retries:
                print(f"    [ERROR] Failed to download {dest_path.name} after {max_retries} attempts: {e}")
                return False
            wait_s = min(60, 3 * attempt)
            print(f"    [RETRY {attempt}/{max_retries}] Network error: {e}. Retrying in {wait_s}s...")
            time.sleep(wait_s)
    return False


def download_open_canopy(
    limit: int | None = None,
    workers: int = 1,
    retries: int = 10,
    token: str | None = None,
) -> bool:
    """
    Bulletproof, pair-by-pair downloader for Open-Canopy 2023.

    Downloads:
      1. canopy_height/geometries.geojson
      2. Pairs: (LiDAR + SPOT) for each of the 18 tiles
    As soon as each pair completes, it immediately hardlinks/symlinks it
    into Open-Canopy/images and Open-Canopy/canopy_height so it is instantly
    usable for training!
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub is not installed.")
        return False

    OC_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 72)
    print("# OPEN-CANOPY 2023 — BULLETPROOF PAIRED DOWNLOADER")
    print("#" * 72)

    # 1. First, download geometries.geojson
    print("\n[Step 1/2] Verifying geometries.geojson...")
    geo_ok = download_open_canopy_file(
        "canopy_height/geometries.geojson",
        max_retries=retries,
        token=token,
    )
    if not geo_ok:
        print("[WARNING] Could not download geometries.geojson; proceeding with tiles...")

    # 2. Query repo metadata to get matching paired tiles
    print("\n[Step 2/2] Resolving 2023 tile pairs from Hugging Face...")
    api = HfApi()
    try:
        info = api.repo_info(OC_REPO, repo_type="dataset", files_metadata=True, token=token)
    except Exception as e:
        print(f"[ERROR] Could not fetch repo metadata from {OC_REPO}: {e}")
        return False

    spot_map = {}
    lidar_map = {}
    for f in info.siblings:
        fn = f.rfilename
        if fn.startswith("canopy_height/2023/spot/") and fn.endswith(".tif"):
            tid = fn.split("/")[-1].replace("compressed_pansharpened_", "").replace(".tif", "")
            spot_map[tid] = f
        elif fn.startswith("canopy_height/2023/lidar/") and fn.endswith(".tif"):
            tid = fn.split("/")[-1].replace("compressed_lidar_", "").replace(".tif", "")
            lidar_map[tid] = f

    common_ids = sorted(set(spot_map.keys()) & set(lidar_map.keys()))
    if limit is not None and limit > 0:
        common_ids = common_ids[:limit]

    total_pairs = len(common_ids)
    print(f"Found {total_pairs} paired tiles to download.")

    update_status("opencanopy_2023", state="downloading", total_pairs=total_pairs)

    completed_pairs = 0
    try:
        for idx, tid in enumerate(common_ids, 1):
            spot_item = spot_map[tid]
            lidar_item = lidar_map[tid]
            total_pair_mb = ((spot_item.size or 0) + (lidar_item.size or 0)) / (1024 ** 2)

            print("\n" + "=" * 72)
            print(f"[Tile Pair {idx}/{total_pairs}] ID: {tid} (~{total_pair_mb / 1024:.2f} GB)")
            print("=" * 72)

            # Download LiDAR (height)
            print(f"  (a) LiDAR Height Map:")
            l_ok = download_open_canopy_file(lidar_item.rfilename, lidar_item.size, max_retries=retries, token=token)

            # Download SPOT (RGB)
            print(f"  (b) SPOT Satellite Image:")
            s_ok = download_open_canopy_file(spot_item.rfilename, spot_item.size, max_retries=retries, token=token)

            if l_ok and s_ok:
                completed_pairs += 1
                # Link immediately
                prepare_open_canopy_layout()
                print(f"  ==> [SUCCESS] Pair {idx}/{total_pairs} ({tid}) is 100% complete and ready to train!")
                update_status("opencanopy_2023", completed_pairs=completed_pairs, current_tile=tid)
            else:
                print(f"  ==> [WARNING] Pair {idx}/{total_pairs} ({tid}) had an error. Continuing to next tile...")

        prepare_open_canopy_layout()
        final = inspect_open_canopy()
        update_status("opencanopy_2023", state="finished", completed_pairs=completed_pairs)
        print("\n" + "=" * 72)
        print(f"[SUMMARY] Finished processing {total_pairs} pairs. Ready on disk: {final['spot_count']} SPOT, {final['lidar_count']} LiDAR.")
        print("=" * 72)
        return completed_pairs == total_pairs

    except KeyboardInterrupt:
        print("\n\n[Open-Canopy] Download paused by user (Ctrl+C).")
        print("All completed files and pairs are preserved and ready.")
        print("Run the command again anytime to resume seamlessly.")
        prepare_open_canopy_layout()
        update_status("opencanopy_2023", state="paused", completed_pairs=completed_pairs)
        return False


# ============================================================================
# DFC2023
# ============================================================================


def dfc_zip_files() -> list[Path]:
    if not DFC_DIR.exists():
        return []

    return sorted(
        DFC_DIR.glob("*.zip")
    )


def inspect_dfc2023() -> dict:
    print("\n" + "=" * 72)
    print("DFC2023 LOCAL STATE")
    print("=" * 72)

    DFC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    size = directory_size(DFC_DIR)

    zips = dfc_zip_files()

    rgb_dir = DFC_DIR / "rgb"
    ndsm_dir = DFC_DIR / "ndsm"

    rgb_count = (
        len(list(rgb_dir.rglob("*")))
        if rgb_dir.exists()
        else 0
    )

    ndsm_count = (
        len(list(ndsm_dir.rglob("*")))
        if ndsm_dir.exists()
        else 0
    )

    print(
        f"Directory:       {DFC_DIR}"
    )

    print(
        f"Local size:      {bytes_to_gb(size):.2f} GB"
    )

    print(
        f"ZIP files:       {len(zips)}"
    )

    for z in zips:
        try:
            zsize = format_bytes(
                z.stat().st_size
            )
        except Exception:
            zsize = "unknown"

        print(
            f"  {z.name}  {zsize}"
        )

    print(
        f"RGB contents:    {rgb_count}"
    )

    print(
        f"nDSM contents:   {ndsm_count}"
    )

    print("=" * 72)

    return {
        "size_bytes": size,
        "zip_count": len(zips),
        "rgb_count": rgb_count,
        "ndsm_count": ndsm_count,
    }


def extract_dfc2023() -> bool:
    """
    Extract DFC ZIP archives.

    Original ZIP files are NEVER deleted.
    """

    DFC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    zips = dfc_zip_files()

    if not zips:
        return False

    for zip_path in zips:

        marker = Path(
            str(zip_path) + ".extracted"
        )

        if marker.exists():
            print(
                f"[DFC2023] Already extracted: "
                f"{zip_path.name}"
            )

            continue

        print(
            "\n[DFC2023] Extracting:"
        )

        print(
            f"  {zip_path}"
        )

        print(
            "Original ZIP will be preserved."
        )

        try:
            # Test ZIP integrity first.
            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as z:

                bad = z.testzip()

                if bad is not None:
                    print(
                        f"[ERROR] ZIP integrity check failed "
                        f"at: {bad}"
                    )

                    return False

                print(
                    f"  Archive members: {len(z.infolist())}"
                )

                z.extractall(
                    DFC_DIR
                )

            marker.write_text(
                "Extraction completed successfully.\n",
                encoding="utf-8",
            )

            print(
                f"[DFC2023] Extracted {zip_path.name}"
            )

        except zipfile.BadZipFile as e:

            print(
                f"[ERROR] Invalid/corrupt ZIP: {e}"
            )

            return False

        except Exception as e:

            print(
                f"[ERROR] Extraction failed: "
                f"{type(e).__name__}: {e}"
            )

            return False

    return True


def classify_dfc_files() -> None:
    """
    Attempt to organize obvious RGB/optical and nDSM/height TIFF files.

    This is deliberately conservative.
    """

    rgb_dir = DFC_DIR / "rgb"
    ndsm_dir = DFC_DIR / "ndsm"

    rgb_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ndsm_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    moved_rgb = 0
    moved_ndsm = 0

    tif_files = list(
        DFC_DIR.rglob("*.tif")
    )

    for tif in tif_files:

        try:
            if rgb_dir in tif.parents:
                continue

            if ndsm_dir in tif.parents:
                continue

        except Exception:
            pass

        name = tif.name.lower()

        target_dir = None

        if (
            "ndsm" in name
            or "height" in name
        ):
            target_dir = ndsm_dir

        elif (
            "rgb" in name
            or "optical" in name
        ):
            target_dir = rgb_dir

        if target_dir is None:
            continue

        target = (
            target_dir
            / tif.name
        )

        if target.exists():
            continue

        try:
            shutil.move(
                str(tif),
                str(target),
            )

            if target_dir == rgb_dir:
                moved_rgb += 1
            else:
                moved_ndsm += 1

        except Exception as e:

            print(
                f"[WARNING] Could not move "
                f"{tif.name}: {e}"
            )

    print(
        f"[DFC2023] Classified "
        f"{moved_rgb} RGB and "
        f"{moved_ndsm} nDSM TIFF(s)."
    )


def handle_dfc2023() -> bool:
    """
    DFC2023 is manually downloaded from IEEE DataPort.

    If no ZIP exists:
        print instructions and return False.

    If ZIP exists:
        test integrity
        extract
        preserve original
    """

    print("\n" + "#" * 72)
    print("# DFC2023 TRACK 2")
    print("#" * 72)

    DFC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    zips = dfc_zip_files()

    if not zips:

        print(
            "\n[DFC2023] No ZIP file detected."
        )

        print(
            "\nDownload the Track 2 TRAIN package from:"
        )

        print(
            f"  {DFC_URL}"
        )

        print(
            "\nThen place the downloaded ZIP directly in:"
        )

        print(
            f"  {DFC_DIR}"
        )

        print(
            "\nAfter that run:"
        )

        print(
            "  python download_all_aux_datasets.py "
            "--dataset dfc2023"
        )

        update_status(
            "dfc2023",
            state="waiting_for_zip",
        )

        return False

    print(
        f"\n[DFC2023] Found {len(zips)} ZIP file(s)."
    )

    for z in zips:
        print(
            f"  {z.name}"
        )

    if not extract_dfc2023():

        print(
            "\n[DFC2023] Extraction did not complete."
        )

        update_status(
            "dfc2023",
            state="extraction_failed",
        )

        return False

    classify_dfc_files()

    final = inspect_dfc2023()

    update_status(
        "dfc2023",
        state="prepared",
        zip_count=final["zip_count"],
        size_gb=bytes_to_gb(
            final["size_bytes"]
        ),
    )

    print(
        "\n[DFC2023] PREPARATION COMPLETE."
    )

    return True


# ============================================================================
# TWO-DATASET QUEUE
# ============================================================================


def run_two_dataset_queue(
    oc_workers: int = 1,
    limit: int | None = None,
    token: str | None = None,
) -> None:
    """
    Run:

        Open-Canopy 2023
             ↓
        DFC2023

    M4Heights is intentionally not called.
    """

    print("\n" + "=" * 72)
    print("DEPTH-WIZARD TWO-DATASET QUEUE")
    print("=" * 72)

    print(
        "\nEnabled:"
    )

    print(
        "  1. Open-Canopy 2023"
    )

    print(
        "  2. DFC2023 Track 2"
    )

    print(
        "\nDisabled:"
    )

    print(
        "  M4Heights"
    )

    print_space_report()

    # ----------------------------------------------------------------------
    # OPEN-CANOPY
    # ----------------------------------------------------------------------

    print("\n\n" + "#" * 72)
    print("# QUEUE 1/2 — OPEN-CANOPY 2023")
    print("#" * 72)

    oc_ok = download_open_canopy(
        limit=limit,
        workers=oc_workers,
        retries=10,
        token=token,
    )

    if not oc_ok:
        print(
            "\n[QUEUE] Open-Canopy did not reach verified completion."
        )

        print(
            "[QUEUE] Existing Open-Canopy files are being preserved."
        )

        print(
            "[QUEUE] Continuing to DFC2023 because it is independent."
        )


        return

    # ----------------------------------------------------------------------
    # DFC
    # ----------------------------------------------------------------------

    print("\n\n" + "#" * 72)
    print("# QUEUE 2/2 — DFC2023")
    print("#" * 72)

    dfc_ok = handle_dfc2023()

    # ----------------------------------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------------------------------

    print("\n\n" + "=" * 72)
    print("FINAL TWO-DATASET STATUS")
    print("=" * 72)

    print(
        f"Open-Canopy 2023: "
        f"{'[COMPLETE]' if oc_ok else '[INCOMPLETE]'}"
    )

    print(
        f"DFC2023:          "
        f"{'[READY]' if dfc_ok else '[WAITING / INCOMPLETE]'}"
    )

    print(
        "M4Heights:        [DISABLED]"
    )

    print(
        "\nStatus file:"
    )

    print(
        f"  {STATUS_FILE}"
    )


# ============================================================================
# CLI
# ============================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Depth-Wizard Open-Canopy 2023 + DFC2023 "
            "safe/resumable downloader."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "opencanopy",
            "dfc2023",
            "two",
        ],
        default="two",
        help=(
            "Operation: opencanopy, dfc2023, or two. "
            "Default: two."
        ),
    )

    parser.add_argument(
        "--oc-workers",
        type=int,
        default=1,
        help=(
            "Concurrent Open-Canopy file downloads. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of tile pairs to download (e.g. --limit 5). Default: all.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face access token.",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect current local data without downloading.",
    )

    parser.add_argument(
        "--check-space",
        action="store_true",
        help="Display disk-space information.",
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================


def main():

    parser = build_parser()

    args = parser.parse_args()

    # Safety: never allow nonsensical worker counts.
    if args.oc_workers < 1:
        print(
            "[ERROR] --oc-workers must be >= 1"
        )
        sys.exit(2)

    # Create directories, but NEVER delete anything.
    OC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DFC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.check_space:
        print_space_report()
        return

    if args.inspect:

        inspect_open_canopy()
        inspect_dfc2023()

        return

    token = args.token or os.environ.get("HF_TOKEN")

    # ----------------------------------------------------------------------
    # SINGLE OPEN-CANOPY
    # ----------------------------------------------------------------------

    if args.dataset == "opencanopy":

        ok = download_open_canopy(
            limit=args.limit,
            workers=args.oc_workers,
            retries=10,
            token=token,
        )

        sys.exit(
            0 if ok else 1
        )

    # ----------------------------------------------------------------------
    # SINGLE DFC
    # ----------------------------------------------------------------------

    if args.dataset == "dfc2023":

        ok = handle_dfc2023()

        sys.exit(
            0 if ok else 1
        )

    # ----------------------------------------------------------------------
    # TWO DATASET QUEUE
    # ----------------------------------------------------------------------

    if args.dataset == "two":

        run_two_dataset_queue(
            oc_workers=args.oc_workers,
            limit=args.limit,
            token=token,
        )

        return


if __name__ == "__main__":
    main()
