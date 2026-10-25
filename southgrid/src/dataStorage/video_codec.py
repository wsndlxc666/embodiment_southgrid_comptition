"""自适应视频编码器选择（SouthGrid 数据管线）。

管线最初把相机视频固定为 ``av1_nvenc``。AV1 NVENC 只在 NVIDIA Ada-Lovelace
（RTX 40 系列及更新）GPU 上可用；在 Ampere（RTX 30 系列及更早）上，codec 描述符
存在，但硬件在 open 时报 ``ExternalError`` 拒绝。与其写死一个编码器，本模块对
实际的 PyAV/FFmpeg 构建 *和* GPU 做真实编码探测，挑出当前机器上真正可用的最优
编码器：

    1. ``av1_nvenc``  — AV1 硬件（RTX 40 系列及更新）
    2. ``hevc_nvenc`` — H.265 硬件（RTX 20 系列及更新）
    3. ``h264_nvenc`` — H.264 硬件（所有支持 NVENC 的 GPU）
    4. ``libsvtav1``  — AV1 软件（需要带 SVT-AV1 的 FFmpeg 构建）
    5. ``libx264``    — H.264 软件（通用兜底）

可用 ``SOUTHGRID_VCODEC`` 环境变量指定编码器名（如 ``libsvtav1``）强制选择。
选择结果按进程缓存。

Note: 这里的探测只验证编码器能开起来；探测尺寸 256×256 与真实分辨率不同，但
编码器能力与尺寸无关，256 对 NVENC（要求 ≥256）是最小可行探测。
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading

_log = logging.getLogger("dataStorage.video_codec")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False

VCODEC_ENV = "SOUTHGRID_VCODEC"

# 首选编码器及各自对应的 PyAV ``st.options``。NVENC 选项对所有 *_nvenc 通用；
# 软件编码器用 ``crf``/``preset``。
_ENCODER_CANDIDATES: list[tuple[str, dict]] = [
    ("av1_nvenc",  {"cq": "30", "preset": "p4"}),
    ("hevc_nvenc", {"cq": "30", "preset": "p4"}),
    ("h264_nvenc", {"cq": "30", "preset": "p4"}),
    ("libsvtav1",  {"crf": "30", "preset": "8"}),
    ("libx264",    {"crf": "30", "preset": "veryfast"}),
]

_LOCK = threading.Lock()
_selected: tuple[str, dict] | None = None


def gpu_summary() -> str:
    """返回一行人类可读的 GPU 摘要；失败时返回占位串。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "N/A (nvidia-smi 不可用)"


def _probe_encode(
    codec: str,
    options: dict,
    *,
    width: int = 256,
    height: int = 256,
    fps: int = 20,
) -> bool:
    """真实编码一帧小图来验证编码器可用（描述符在缺少硬件时会撒谎）。"""
    import av
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        container = av.open(tmp, "w")
        try:
            st = container.add_stream(codec, rate=fps)
            st.width = width
            st.height = height
            st.pix_fmt = "yuv420p"
            st.options = options
            frame = av.VideoFrame(width, height, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for pkt in st.encode(frame):
                container.mux(pkt)
            for pkt in st.encode():
                container.mux(pkt)
        finally:
            container.close()
        return True
    except Exception as e:
        _log.debug("codec %r 探测失败: %s", codec, e)
        return False
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def select_video_codec() -> tuple[str, dict]:
    """解析要使用的视频编码器，首次调用时对硬件做真实探测。

    返回 ``(codec_name, stream_options)``。若没有任何编码器能真正打开，抛出
    RuntimeError（此时环境确实不可用）。
    """
    global _selected
    if _selected is not None:
        return _selected

    with _LOCK:
        if _selected is not None:
            return _selected

        override = os.environ.get(VCODEC_ENV, "").strip()
        if override:
            opts = dict(next((o for c, o in _ENCODER_CANDIDATES if c == override), {}))
            if not _probe_encode(override, opts):
                raise RuntimeError(
                    f"{VCODEC_ENV}={override!r} 已设置，但该编码器在本机无法打开"
                    f"（GPU {gpu_summary()}）。请移除该变量，或从以下编码器中选择: "
                    + ", ".join(c for c, _ in _ENCODER_CANDIDATES)
                )
            _selected = (override, opts)
            _log.info("[视频] 使用指定编码器 %s", override)
            return _selected

        tried: list[str] = []
        for codec, opts in _ENCODER_CANDIDATES:
            if _probe_encode(codec, opts):
                _selected = (codec, opts)
                kind = "硬件" if codec.endswith("_nvenc") else "软件"
                _log.info("[视频] 选择编码器 %s（%s编码）", codec, kind)
                return _selected
            tried.append(codec)

        raise RuntimeError(
            "未找到可用的视频编码器。已尝试: " + ", ".join(tried)
            + f"。GPU: {gpu_summary()}。请检查 PyAV/FFmpeg 与 NVIDIA 驱动。"
        )
