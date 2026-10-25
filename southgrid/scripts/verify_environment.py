#!/usr/bin/env python3
"""Fail-fast verification for the pinned SouthGrid delivery runtime."""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys
import tempfile

# The pinned runtime packages live in src/dataStorage (imported below for the
# shared video-codec check); make the source tree importable from any cwd.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

def _assert_version(distribution: str, expected: str) -> None:
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(f"{distribution}: expected {expected}, got {actual}")


def _assert_environment_owned(module) -> None:
    module_path = pathlib.Path(module.__file__).resolve()
    environment_root = pathlib.Path(sys.prefix).resolve()
    if not module_path.is_relative_to(environment_root):
        raise RuntimeError(
            f"{module.__name__} loaded outside the active environment: {module_path}"
        )


def main() -> None:
    if sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError(f"expected Python 3.12.13, got {sys.version.split()[0]}")

    expected_versions = {
        "numpy": "2.2.6",
        "scipy": "1.16.2",
        "scipy-stubs": "1.16.2.0",
        "numpy-typing-compat": "20250818.2.2",
        "optype": "0.13.4",
        "orca-gym": "26.7.3",
        "orca-lab": "26.7.3",
        "gymnasium": "1.2.1",
        "mujoco": "3.7.0",
        "av": "17.0.1",
        "pyarrow": "24.0.0",
        "opencv-python": "4.13.0.92",
        "torch": "2.7.1+cpu",
        "torchvision": "0.22.1+cpu",
        "datasets": "3.6.0",
        "lerobot": "0.3.4+orca.1",
        "televuer": "4.0.0+orca.1",
        "openpi-client": "0.1.0+orca.1",
        "aiohttp-cors": "0.8.1",
        "argcomplete": "3.7.0",
        "dm-tree": "0.1.10",
        "expandvars": "1.1.2",
        "killport": "1.2.0",
        "msgpack": "1.1.2",
        "params-proto": "2.13.2",
        "psutil": "7.1.2",
        "vuer": "0.0.60",
        "waterbear": "2.6.8",
        "wrapt": "2.2.1",
    }
    for distribution, expected in expected_versions.items():
        _assert_version(distribution, expected)

    numpy_distributions = [
        dist
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"].lower().replace("_", "-") == "numpy"
    ]
    if len(numpy_distributions) != 1:
        raise RuntimeError(f"expected one NumPy distribution, found {len(numpy_distributions)}")

    import cv2
    import psutil
    import tree
    import lerobot
    import televuer
    import openpi_client
    from openpi_client import msgpack_numpy

    if tree.map_structure(lambda value: value + 1, {"value": 1}) != {"value": 2}:
        raise RuntimeError("dm-tree runtime check failed")
    if psutil.Process().pid <= 0:
        raise RuntimeError("psutil runtime check failed")

    gui_line = next(
        (line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line),
        "",
    )
    if "QT5" not in gui_line:
        raise RuntimeError(f"OpenCV GUI build required for eval preview, got: {gui_line!r}")

    # Verify the complete camera-pipeline import chain.
    import orca_gym.environment  # noqa: F401
    from orca_gym.sensor.rgbd_camera import CameraWrapper  # noqa: F401

    # Verify the OrcaLab and OrcaGym version contract.
    _orca_lab_pin = next(
        (
            requirement
            for requirement in importlib.metadata.requires("orca-lab") or ()
            if requirement.startswith("orca-gym==")
        ),
        None,
    )
    _orca_gym_version = importlib.metadata.version("orca-gym")
    if _orca_lab_pin != f"orca-gym=={_orca_gym_version}":
        raise RuntimeError(
            f"orca-lab requires {_orca_lab_pin!r}, but orca-gym {_orca_gym_version} is installed"
        )

    # Import the launcher entry point to verify that the desktop application is available.
    import orcalab.launcher  # noqa: F401

    import numpy as np

    # Verify the pipeline's video encoder with an actual encode, not just a
    # codec descriptor lookup (the descriptor exists even when the GPU lacks
    # the hardware encoder).  Selection is adaptive: av1_nvenc on RTX 40-series
    # or newer, falling back to H.265/H.264 NVENC or software encoders on older
    # GPUs.  See dataStorage.video_codec for the candidate order and the
    # SOUTHGRID_VCODEC override.
    from dataStorage.video_codec import gpu_summary, select_video_codec

    _video_codec, _video_codec_options = select_video_codec()
    _gpu_line = gpu_summary()

    payload = {"state": [1.0, 2.0], "image": np.zeros((2, 2, 3), dtype="uint8")}
    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    if restored["image"].shape != (2, 2, 3):
        raise RuntimeError("OpenPI msgpack NumPy round-trip failed")

    # Verify data-only episode serialization.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with tempfile.TemporaryDirectory(prefix="orca_lerobot_verify_") as tmp:
        root = pathlib.Path(tmp) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id="orca/runtime-selfcheck",
            fps=20,
            root=root,
            robot_type="runtime-selfcheck",
            features={
                "observation.state": {"dtype": "float32", "shape": (2,)},
                "action": {"dtype": "float32", "shape": (2,)},
            },
        )
        dataset.add_frame(
            {
                "observation.state": np.array([0.0, 1.0], dtype=np.float32),
                "action": np.array([1.0, 2.0], dtype=np.float32),
            },
            task="runtime selfcheck",
        )
        episode_index = dataset.save_episode_data_only()
        parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
        if episode_index != 0 or not parquet.is_file():
            raise RuntimeError("LeRobot data-only episode write failed")

    print("Environment verification OK")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  GPU: {_gpu_line}")
    print(f"  Video encoder: {_video_codec} (options {_video_codec_options})")
    print(f"  OpenCV: {gui_line.strip()}")
    print(f"  OrcaLab/OrcaGym: {importlib.metadata.version('orca-lab')} / {_orca_gym_version}")
    print("  LeRobot/TeleVuer/OpenPI client: versions verified")


if __name__ == "__main__":
    main()
