"""LeRobot dataset storage with streaming NVENC video encoding.

Frames are sampled on the configured wall or simulation clock. Each emitted frame
pairs the previous observation with the next state-derived action, then streams
camera images to the selected encoder backend while buffering dataset metadata.
Successful episodes flush MP4, parquet, statistics, and metadata before returning.

State and action schemas are defined by each robot-specific subclass. By default,
``action[i] = state[i+1]``; subclasses may provide another documented action schema.
The validated runtime uses LeRobot 0.3.x, OrcaGym 26.7.3, PyAV, and PyArrow.
"""
from __future__ import annotations

import av
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import cv2

from dataStorage.lerobot_camera import (
    camera_keys,
    capture_frame_with_idx,
    iter_frames_from_mp4,
)
from dataStorage.video_codec import select_video_codec
from dataStorage.openloong_data_storage import OpenLoongDataStorage
from dataStorage.tiangong_data_storage import Tiangong2DataStorage
from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


# ---------------------------------------------------------------------------
# 模块日志器
# ---------------------------------------------------------------------------
# Use a module logger by default; entry points may inject their configured logger.
_log = logging.getLogger("dataStorage.lerobot")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False


def set_logger(logger) -> None:
    """让本模块的日志并入调用方的 logger（例如采集脚本的 orca_logger）。"""
    global _log
    if logger is not None:
        _log = logger


# ---------------------------------------------------------------------------
# 动态导入 lerobot（仅在 orcalab_lerobot 环境中可用）
# ---------------------------------------------------------------------------

def _import_lerobot_dataset():
    for mod in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            import importlib
            return importlib.import_module(mod).LeRobotDataset
        except Exception:
            pass
    raise ImportError(
        "无法导入 LeRobotDataset。请在 orcalab_lerobot 环境中运行，"
        "并确认已安装 lerobot>=0.3.0。"
    )


# LeRobot 自动补的列（meta/DEFAULT_FEATURES）。resume 时拿来区分"自带的"与"我们加的"。
DEFAULT_LEROBOT_FEATURES = {
    "timestamp", "frame_index", "episode_index", "index", "task_index",
}


def _import_video_encoding_manager():
    for mod in (
        "lerobot.datasets.video_utils",
        "lerobot.common.datasets.video_utils",
    ):
        try:
            import importlib
            return importlib.import_module(mod).VideoEncodingManager
        except Exception:
            pass
    raise ImportError("无法导入 VideoEncodingManager，请检查 lerobot 版本。")


# ---------------------------------------------------------------------------
# 流式视频编码（自适应编码器，内存 numpy → mp4，无 PNG 磁盘往返）
# ---------------------------------------------------------------------------

class _CameraEncodeWorker:
    """单相机后台编码线程：RGB numpy 帧 → 自适应编码器 → mp4（PyAV）。"""

    _FINISH = object()   # 正常结束哨兵
    _DISCARD = object()  # 丢弃哨兵

    def __init__(self, video_path: Path, fps: int, width: int, height: int) -> None:
        self._path = video_path
        self._fps = fps
        self._width = width
        self._height = height
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._ready = threading.Event()
        self._open_error: str | None = None
        self._thread = threading.Thread(
            target=self._worker, daemon=True,
            name=f"nvenc_{video_path.stem}"
        )
        self._thread.start()

    @property
    def wh(self) -> tuple:
        return (self._width, self._height)

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Wait until the video encoding session is ready.
        """
        if not self._ready.wait(timeout):
            raise TimeoutError("编码会话打开超时")
        if self._open_error:
            raise RuntimeError("编码会话打开失败")

    def push(self, np_rgb: np.ndarray) -> None:
        """推入一帧 RGB uint8 HWC numpy 数组。队列满时丢帧，不阻塞控制线程。"""
        try:
            self._q.put_nowait(np_rgb)
        except queue.Full:
            pass

    def finish(self) -> None:
        """刷新编码器并关闭 mp4，同步等待线程退出。"""
        self._q.put(self._FINISH)
        self._thread.join()

    def discard(self) -> None:
        """丢弃所有帧，删除半成品文件，同步等待线程退出。"""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(self._DISCARD)
        self._thread.join()

    def _worker(self) -> None:
        try:
            codec, codec_options = select_video_codec()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(self._path), "w")
            st = container.add_stream(codec, rate=self._fps)
            st.pix_fmt = "yuv420p"
            st.width = self._width
            st.height = self._height
            st.options = codec_options
            # Open the codec before accepting episode frames.
            st.codec_context.open()
        except BaseException as e:
            self._open_error = repr(e)
            self._ready.set()
            raise
        self._ready.set()

        frame_idx = 0
        discard = False
        try:
            while True:
                item = self._q.get()
                if item is self._FINISH:
                    break
                if item is self._DISCARD:
                    discard = True
                    break
                av_frame = av.VideoFrame.from_ndarray(item, format="rgb24")
                av_frame.pts = frame_idx
                frame_idx += 1
                pkt = st.encode(av_frame)
                if pkt:
                    container.mux(pkt)
        finally:
            if not discard:
                try:
                    pkt = st.encode()
                    if pkt:
                        container.mux(pkt)
                except Exception:
                    pass
            container.close()
            if discard and self._path.exists():
                try:
                    self._path.unlink()
                except Exception:
                    pass


class StreamingNvencEncoder:
    """每相机一路流式编码（自适应编码器，GPU 优先），零 PNG 磁盘往返。

    生命周期（每集）：
        push(cam_key, np_rgb)  — 推入帧（首帧惰性按尺寸初始化 worker）
        end_episode()          — 同步 flush + 关闭 mp4 + 清理空目录
        discard_episode()      — 丢弃本集 + 删除半成品 + 清理空目录
        close()                — 退出清理（有进行中的集则 discard）
    """

    def __init__(self, dataset, fps: int) -> None:
        self._dataset = dataset
        self._fps = fps
        self._workers: dict[str, _CameraEncodeWorker] = {}
        self._ep_idx: int | None = None

    def _ensure_ep_idx(self) -> None:
        """从 dataset.episode_buffer 惰性获取当前 episode_index。"""
        if self._ep_idx is None:
            self._ep_idx = self._dataset.episode_buffer["episode_index"]

    def push(self, cam_key: str, np_rgb: np.ndarray, jpeg_path=None) -> bool:
        """推入一帧（RGB HWC uint8），首帧时惰性启动 worker。

        jpeg_path 由 subproc 后端使用；本 inproc 路径忽略该参数，仅保持签名一致。
        返回 True 表示帧已入队（或已计数丢弃）；接口与 EncoderProcClient.push 对齐。
        """
        self._ensure_ep_idx()
        h, w = np_rgb.shape[:2]
        worker = self._workers.get(cam_key)
        if worker is not None and worker.wh != (w, h):
            # Recreate the session when the incoming frame size changes.
            _log.warning(
                f"[视频] {cam_key} 分辨率已变化，正在更新编码会话"
            )
            worker.discard()
            worker = None
        if worker is None:
            worker = self._spawn_worker(cam_key, h, w)
            self._workers[cam_key] = worker
        worker.push(np_rgb)
        return True

    def _spawn_worker(self, cam_key: str, height: int, width: int
                      ) -> _CameraEncodeWorker:
        feat_key = f"observation.images.{cam_key}"
        video_path = (
            Path(str(self._dataset.root))
            / self._dataset.meta.get_video_file_path(self._ep_idx, feat_key)
        )
        return _CameraEncodeWorker(video_path, self._fps, int(width), int(height))

    def start_episode(self, ep_idx: int, height: int, width: int) -> float:
        """Initialize all camera encoding sessions before episode capture.

        Returns the initialization duration in seconds.
        """
        self._ep_idx = int(ep_idx)
        t0 = time.perf_counter()
        for feat_key in self._dataset.meta.video_keys:
            cam_key = feat_key[len("observation.images."):]
            if cam_key not in self._workers:
                self._workers[cam_key] = self._spawn_worker(cam_key, height, width)
        for w in self._workers.values():
            w.wait_ready()
        dt = time.perf_counter() - t0
        _log.info("[视频] 本集编码会话已就绪")
        return dt

    def prewarm(self, cam_keys: list[str], height: int, width: int) -> float:
        """Initialize the process-level encoder context.

        Returns the initialization duration in seconds.
        """
        import tempfile

        t0 = time.perf_counter()
        h, w = int(height), int(width)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for cam_key in cam_keys:
            tmp = Path(tempfile.gettempdir()) / f"southgrid_encoder_check_{cam_key}_{os.getpid()}.mp4"
            worker = None
            try:
                worker = _CameraEncodeWorker(tmp, self._fps, w, h)
                worker.push(dummy)
                worker.finish()
            except Exception as e:
                _log.warning(f"[视频] {cam_key} 编码器预初始化失败")
                if worker is not None:
                    try:
                        worker.discard()
                    except Exception:
                        pass
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
        dt = time.perf_counter() - t0
        _log.info("[视频] 编码器预初始化完成")
        return dt

    def end_episode(self) -> None:
        """刷新所有相机 mp4，同步等待。保留 ep_idx 供后续统计 PNG 写入和清理使用。"""
        for w in self._workers.values():
            w.finish()
        self._workers = {}
        # ep_idx 保留，等 cleanup_episode() 清理

    def cleanup_episode(self) -> None:
        """Clean episode image files and reset the episode index."""
        self._cleanup_tmp_dirs()
        self._ep_idx = None

    def discard_episode(self) -> None:
        """Discard the current episode and remove its incomplete files."""
        for w in self._workers.values():
            w.discard()
        self._cleanup_tmp_dirs()
        self._workers = {}
        self._ep_idx = None

    def _cleanup_tmp_dirs(self) -> None:
        """Remove episode image files after finalization."""
        if self._ep_idx is None:
            return
        try:
            for vk in self._dataset.meta.video_keys:
                img_dir = self._dataset._get_image_file_path(
                    episode_index=self._ep_idx, image_key=vk, frame_index=0
                ).parent
                if img_dir.exists():
                    shutil.rmtree(img_dir, ignore_errors=True)
        except Exception:
            pass

    def close(self) -> None:
        """退出清理：如有进行中集则丢弃。"""
        if self._workers:
            self.discard_episode()
        elif self._ep_idx is not None:
            self._cleanup_tmp_dirs()
            self._ep_idx = None


# ---------------------------------------------------------------------------
# Statistics-image writer with a bounded queue.
# ---------------------------------------------------------------------------

class _StatsImageWriter:
    """Write statistics JPEGs through a bounded background queue.

    ``wait_until_done`` flushes pending images, ``drop_pending`` discards an
    unfinished episode, and ``stop`` terminates the worker threads.
    """

    _STOP = object()

    def __init__(self, maxsize: int = 128, num_threads: int = 3,
                 jpeg_quality: int = 95) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._jpeg_quality = jpeg_quality
        self._threads: list[threading.Thread] = []
        self._stopped = False
        for i in range(num_threads):
            t = threading.Thread(
                target=self._worker, daemon=True,
                name=f"stats_jpg_w{i}"
            )
            t.start()
            self._threads.append(t)

    def push(self, image: np.ndarray, fpath) -> None:
        """把 HWC RGB uint8 图像异步写为 JPEG，满时丢帧不阻塞主线程。"""
        try:
            self._q.put_nowait((image, fpath))
        except queue.Full:
            pass

    def wait_until_done(self) -> None:
        """阻塞直到队列中所有条目都被处理完（用于 flush 前保证文件已落盘）。"""
        self._q.join()

    def drop_pending(self) -> None:
        """丢弃队列中尚未取走的所有条目；等在途写操作完成后返回。"""
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break
        self._q.join()

    def stop(self) -> None:
        """停止所有 worker 线程（发哨兵 + join）。幂等。"""
        if self._stopped:
            return
        for _ in self._threads:
            self._q.put(self._STOP)
        for t in self._threads:
            t.join()
        self._stopped = True

    def _worker(self) -> None:
        quality_flag = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        # Match the image downsampling used by LeRobot episode statistics.
        _ds_thresh = 300
        _ds_target = 150
        while True:
            item = self._q.get()
            try:
                if item is self._STOP:
                    return
                image, fpath = item
                fpath = Path(fpath)
                fpath.parent.mkdir(parents=True, exist_ok=True)
                # 整数步长降采样（对齐 lerobot auto_downsample_height_width）
                if image.ndim == 3:
                    h, w = image.shape[:2]
                    if max(h, w) >= _ds_thresh:
                        factor = int(w / _ds_target) if w > h else int(h / _ds_target)
                        if factor > 1:
                            image = image[::factor, ::factor]
                    # cv2.imwrite 需要 BGR，源图是 RGB HWC
                    bgr = image[:, :, ::-1]
                else:
                    bgr = image
                ok = cv2.imwrite(str(fpath), bgr, quality_flag)
                if not ok:
                    _log.warning("[LeRobot] 图像统计文件写入失败")
            except Exception as e:
                _log.warning("[LeRobot] 图像统计文件写入失败")
            finally:
                self._q.task_done()


# ---------------------------------------------------------------------------
# LeRobotDatasetWriter：封装 LeRobotDataset 的创建/写入/生命周期（NVENC 直编版）
# ---------------------------------------------------------------------------

class LeRobotDatasetWriter:
    """封装 LeRobotDataset 的创建、NVENC 流式帧写入与 episode 存盘生命周期。

    使用方式：
        writer = LeRobotDatasetWriter.create(...)
        with writer:
            ...  # 调用 stream_frame() / flush_episode() / discard_episode()

    保存流程（每集）：
        1. stream_frame()：图像推 StreamingNvencEncoder 队列（GPU），state/action 进内存缓冲。
        2. flush_episode()：
           a. nvenc_enc.end_episode()  — GPU 编码完成，mp4 直接落盘（通常 <1s）。
           b. save_episode_data_only() — 写 parquet + meta（同步）。
           c. encode_episode_videos()  — 文件已存在跳过；ep0 写 info.json。
    """

    def __init__(
        self,
        dataset,
        nvenc_enc,
        stats_writer: "_StatsImageWriter | None",
        *,
        encode_backend: str = "inproc",
        extra_feature_names: set | None = None,
    ) -> None:
        self._dataset = dataset
        self._nvenc_enc = nvenc_enc
        self._stats_writer = stats_writer
        self._encode_backend = str(encode_backend)
        self._jpeg_in_child = self._encode_backend == "subproc"
        self._saved_episodes: int = 0
        self._frame_idx: int = 0
        self._extra_feature_names = set(extra_feature_names or ())

    def has_feature(self, name: str) -> bool:
        """本数据集是否声明了该额外列（逐帧写帧前用它决定要不要带上）。"""
        return name in self._extra_feature_names

    @classmethod
    def create(
        cls,
        repo_id: str,
        root: str,
        fps: int,
        camera_map: dict,
        state_dim: int,
        state_names: list[str],
        cam_shape: tuple,
        resume: bool = False,
        robot_type: str = "humanoid",
        action_names: list[str] | None = None,
        action_dim: int | None = None,
        encode_backend: str = "inproc",
        enc_ring_slots: int = 96,
        extra_features: dict | None = None,
    ) -> "LeRobotDatasetWriter":
        """创建（或恢复）一个 LeRobotDataset，返回已包装的 NVENC writer。

        action_names / action_dim 默认与 state 相同（绝对 next-step）。
        g1_pick 等 Δq 数据集应显式传入不同的 action_names。

        extra_features: 额外的标量列，如 {"tool_id": {"dtype": "int64", "shape": (1,)}}。
            **逐帧写帧时必须每帧都带上**（LeRobot 的 validate_frame 要求
            features 与 frame 的键严格一一对应），所以只在确实会写它的
            数据集上开。shape 必须写 tuple：list 会落到 Sequence 分支。

        encode_backend:
            "inproc"  — 同进程线程编码
            "subproc" — 独立 forkserver 子进程编码与 JPEG 写入
        """
        backend = str(encode_backend or "inproc").strip().lower()
        if backend not in ("inproc", "subproc"):
            raise ValueError(
                f"encode_backend 必须是 'inproc' 或 'subproc'，收到: {encode_backend!r}"
            )

        LeRobotDataset = _import_lerobot_dataset()
        cams = camera_keys(camera_map)
        act_names = list(action_names) if action_names is not None else list(state_names)
        act_dim = int(action_dim) if action_dim is not None else int(state_dim)
        if act_dim != len(act_names):
            raise ValueError(
                f"action_dim={act_dim} 与 len(action_names)={len(act_names)} 不一致"
            )
        if int(state_dim) != len(state_names):
            raise ValueError(
                f"state_dim={state_dim} 与 len(state_names)={len(state_names)} 不一致"
            )

        if resume and Path(root).exists():
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                download_videos=False,
                tolerance_s=0.0001,
            )
            # Resume only when stored feature names match the requested schema.
            def _flat_names(raw) -> list[str]:
                if raw is None:
                    return []
                if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
                    return [str(x) for x in raw[0]]
                return [str(x) for x in list(raw)]

            try:
                feats = dataset.features
                prev_state_names = _flat_names(feats["observation.state"]["names"])
                prev_action_names = _flat_names(feats["action"]["names"])
                if prev_state_names != list(state_names) or prev_action_names != act_names:
                    raise ValueError(
                        "[resume] 数据集的 state/action 字段与当前配置不一致。"
                        "请使用新的 --lerobot_out，或确认后重新创建目标数据集。"
                    )
                want_extra = set(extra_features or {})
                have_extra = set(feats) - {"observation.state", "action"} - {
                    f"observation.images.{c}" for c in cams
                } - set(DEFAULT_LEROBOT_FEATURES)
                if want_extra != have_extra:
                    raise ValueError(
                        f"[resume] 额外列不一致：数据集有 {sorted(have_extra)}，"
                        f"本次要求 {sorted(want_extra)}。请换 --lerobot_out。"
                    )
            except KeyError as e:
                raise ValueError(f"[resume] 数据集缺少 feature 字段: {e}") from e
            dataset.episode_buffer = dataset.create_episode_buffer()
            print(
                f"[resume] 已加载 {dataset.num_episodes} 集 / {dataset.num_frames} 帧"
            )
        else:
            if Path(root).exists() and not resume:
                shutil.rmtree(root)

            features: dict = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [list(state_names)],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (act_dim,),
                    "names": [act_names],
                },
            }
            for cam_key in cams:
                features[f"observation.images.{cam_key}"] = {
                    "dtype": "video",
                    "shape": cam_shape,
                    "names": ["channels", "height", "width"],
                }
            for k, ft in (extra_features or {}).items():
                if "/" in k:
                    raise ValueError(f"feature 名不能含 '/'：{k}")
                features[k] = dict(ft)

            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(fps),
                robot_type=robot_type,
                features=features,
                root=root,
                use_videos=True,
                tolerance_s=0.0001,
                image_writer_processes=0,
                image_writer_threads=0,
            )

        # NVENC 直接写 mp4，不需要 AsyncImageWriter。
        try:
            dataset.stop_image_writer()
        except Exception:
            pass
        dataset.image_writer = None

        # 两个后端都把图像路径后缀改为 .jpg（add_frame 往 buffer 写路径字符串）
        _orig_img_path = dataset._get_image_file_path
        dataset._get_image_file_path = (
            lambda episode_index, image_key, frame_index: (
                _orig_img_path(
                    episode_index=episode_index,
                    image_key=image_key,
                    frame_index=frame_index,
                ).with_suffix(".jpg")
            )
        )

        if backend == "subproc":
            from dataStorage.encoder_proc import EncoderProcClient

            # cam_shape = (C, H, W)
            h = int(cam_shape[1])
            w = int(cam_shape[2])
            nvenc_enc = EncoderProcClient(
                dataset,
                int(fps),
                cam_keys=list(cams),
                height=h,
                width=w,
                ring_slots=int(enc_ring_slots),
                jpeg_quality=95,
            )
            stats_writer = None
            dataset._save_image = lambda image, fpath: None
            _log.info("[LeRobot] 视频编码服务已启用")
        else:
            stats_writer = _StatsImageWriter(maxsize=128, num_threads=1, jpeg_quality=95)
            dataset._save_image = lambda image, fpath: stats_writer.push(image, fpath)
            nvenc_enc = StreamingNvencEncoder(dataset, int(fps))
            _log.info("[LeRobot] 视频编码器已启用")

        extra_names = set(extra_features or {})
        return cls(
            dataset, nvenc_enc, stats_writer,
            encode_backend=backend, extra_feature_names=extra_names,
        )

    def __enter__(self) -> "LeRobotDatasetWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exc_type, exc_val, exc_tb)
        return False

    def close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """关闭 NVENC 编码器和写盘器（秒级退出）。幂等。"""
        if self._stats_writer is not None:
            try:
                self._stats_writer.stop()
            except Exception:
                pass
        try:
            self._nvenc_enc.close()
        except Exception:
            pass

    def prepare_episode(self, height: int, width: int) -> float:
        """Initialize the episode encoding session before capture begins."""
        fn = getattr(self._nvenc_enc, "start_episode", None)
        if not callable(fn):
            return 0.0
        try:
            ep_idx = int(self._dataset.meta.total_episodes)
            return float(fn(ep_idx, int(height), int(width)) or 0.0)
        except Exception as e:
            _log.warning("[视频] 预初始化不可用，将在接收首帧时创建编码会话")
            return 0.0

    def stream_frame(self, frame: dict, task: str) -> None:
        """Stream camera images and append state/action data for one frame."""
        import time as _time
        _t0 = _time.perf_counter()
        for k, v in frame.items():
            if k.startswith("observation.images."):
                cam_key = k[len("observation.images."):]
                jpg = None
                if self._jpeg_in_child:
                    jpg = self._dataset._get_image_file_path(
                        episode_index=self._dataset.episode_buffer["episode_index"],
                        image_key=k,
                        frame_index=self._frame_idx,
                    )
                self._nvenc_enc.push(cam_key, v, jpeg_path=jpg)
        _t1 = _time.perf_counter()
        self._dataset.add_frame(frame, task)
        _t2 = _time.perf_counter()
        self._frame_idx += 1

        _nvenc_ms = (_t1 - _t0) * 1000.0
        _add_ms   = (_t2 - _t1) * 1000.0
        if _nvenc_ms > 30.0 or _add_ms > 30.0:
            _log.warning("[LeRobot] 当前帧写入耗时较长")

    def flush_episode(self) -> int:
        """Finalize video, parquet, statistics, and episode metadata."""
        # subproc 子进程若已死，拒绝写出缺 mp4 的坏集
        if getattr(self._nvenc_enc, "is_dead", False):
            raise RuntimeError(
                "[LeRobot] 编码子进程已死亡，拒绝 flush_episode。"
                "请 discard 本集并重跑。"
            )

        _log.info("[LeRobot] 结束本集 GPU 编码，等待 mp4 落盘…")
        t0 = time.perf_counter()
        self._nvenc_enc.end_episode()
        t_enc = time.perf_counter() - t0

        t_jpg = 0.0
        if self._stats_writer is not None:
            # inproc：必须先排空写盘队列，再计算 stats
            _log.info(
                "[LeRobot] 视频已写入，正在完成图像统计…"
            )
            t0 = time.perf_counter()
            self._stats_writer.wait_until_done()
            t_jpg = time.perf_counter() - t0
        else:
            # subproc：END_EP ack 已保证 JPEG 落盘
            _log.info(
                "[LeRobot] 视频与图像统计已写入"
            )

        _log.info("[LeRobot] 正在写入回合数据与元信息…")
        t0 = time.perf_counter()
        ep_idx = self._dataset.save_episode_data_only()
        t_meta = time.perf_counter() - t0

        # Statistics images are no longer needed after episode finalization.
        self._nvenc_enc.cleanup_episode()

        _log.info(f"✓  [LeRobot] Episode {ep_idx} 已保存")
        # mp4 已存在故跳过编码；ep0 需调用以写 info.json（update_video_info）
        self._dataset.encode_episode_videos(ep_idx)
        self._saved_episodes += 1
        self._frame_idx = 0
        return ep_idx

    def discard_episode(self) -> None:
        """丢弃本集缓存（帧数不足时调用）。"""
        if self._stats_writer is not None:
            self._stats_writer.drop_pending()
        self._nvenc_enc.discard_episode()
        self._dataset.clear_episode_buffer()
        self._dataset.episode_buffer = self._dataset.create_episode_buffer()
        self._frame_idx = 0

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames


# ---------------------------------------------------------------------------
# LeRobotSimSyncMixin：时钟门控流式写帧 + episode 管理
# ---------------------------------------------------------------------------

class LeRobotSimSyncMixin:
    """注入到 DataStorage 子类中，覆盖 collection_data / save_data / clear_data。

    使用前必须先调用 configure_lerobot(...)，传入相机流句柄和 writer。
    子类必须实现：
        build_state(obs: dict) -> np.ndarray
        state_dim: int property
        state_names: list[str] property
    可选覆盖：
        build_action(state_prev, state_cur) -> np.ndarray
            默认返回 state_cur（绝对 next-step）；g1_pick 覆盖为 Δq。
    """

    def configure_lerobot(
        self,
        fps: float,
        cameras: dict,
        camera_map: dict,
        target_hw: tuple,
        writer: LeRobotDatasetWriter,
        task: str = "robot manipulation",
        clock: str = "sim",
        camera_source: str = "websocket",
    ) -> None:
        if clock not in ("sim", "wall"):
            raise ValueError(f"clock 只能是 'sim' 或 'wall'，收到: {clock!r}")
        if camera_source not in ("websocket", "mp4"):
            raise ValueError(f"camera_source 只能是 'websocket' 或 'mp4'，收到: {camera_source!r}")
        self._lr_fps = float(fps)
        self._lr_clock = clock
        self._lr_camera_source = camera_source
        self._lr_cameras = cameras
        self._lr_camera_map = camera_map
        self._lr_target_hw = target_hw
        self._lr_writer = writer
        self._lr_task = task
        self._lr_tool_src = None
        self._reset_episode()

    def set_task(self, task: str) -> None:
        """更新后续录制帧写入 LeRobot 的语言指令（逐集可变）。空串忽略。"""
        if task and task.strip():
            self._lr_task = task.strip()

    def set_tool_id_source(self, fn) -> None:
        """注册"当前控制步的工具标签"取值回调（返回 int，-1=撤离/等待）。

        标签由采集器的设备在每个控制步发布；本 mixin 按与 state 相同的
        **一帧滞后**写进数据集，保证 tool_id 与同一帧的 state/images 对齐。
        数据集没有声明 tool_id 列时这个回调不会被调用。
        """
        self._lr_tool_src = fn

    def build_state(self, obs: dict) -> np.ndarray:
        raise NotImplementedError("子类必须实现 build_state(obs)")

    def build_action(
        self, state_prev: np.ndarray, state_cur: np.ndarray
    ) -> np.ndarray:
        """由相邻两帧 state 构造 action。默认绝对 next-step（= state_cur）。"""
        return np.asarray(state_cur, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        raise NotImplementedError("子类必须实现 state_dim property")

    @property
    def state_names(self) -> list[str]:
        raise NotImplementedError("子类必须实现 state_names property")

    @property
    def buffered_frame_count(self) -> int:
        """当前集已经过时钟门控的次数，在 save_data/clear_data 前读取。"""
        return self._lr_count

    def collection_data(self, obs: dict, env: "OrcaGymLocalEnv", **kwargs) -> None:
        """时钟门控流式写帧（websocket）或仅记录 state/时间戳（mp4）。"""
        t = time.perf_counter() if self._lr_clock == "wall" else float(env.data.time)
        if self._lr_next_cap is None:
            self._lr_next_cap = t

        if t + 1e-9 < self._lr_next_cap:
            return

        if self._lr_camera_source == "mp4":
            state_cur = self.build_state(obs)
            wall_t = time.perf_counter()
            if self._lr_ep_start_wall is None:
                self._lr_ep_start_wall = wall_t
            self._lr_states.append((state_cur, wall_t))
            self._lr_count += 1
            self._lr_next_cap += 1.0 / self._lr_fps
            return

        state_cur = self.build_state(obs)
        _cam_t0 = time.perf_counter()
        images_cur, cam_indices_cur = capture_frame_with_idx(
            self._lr_cameras, self._lr_camera_map, self._lr_target_hw
        )
        _cam_ms = (time.perf_counter() - _cam_t0) * 1000.0
        if _cam_ms > 20.0:
            _log.warning("[相机] 当前帧读取耗时较长")

        want_tool = self._lr_tool_src is not None and self._lr_writer.has_feature("tool_id")
        tool_cur = int(self._lr_tool_src()) if want_tool else -1

        if self._lr_prev is not None:
            state_prev, images_prev, _ = self._lr_prev
            cams = camera_keys(self._lr_camera_map)
            frame: dict = {
                "observation.state": state_prev.astype(np.float32),
                "action": self.build_action(state_prev, state_cur).astype(np.float32),
            }
            for cam_key in cams:
                frame[f"observation.images.{cam_key}"] = images_prev[cam_key]
            if want_tool:
                # 与 state/images 同源同滞后：写的是**上一控制步**的标签
                frame["tool_id"] = np.asarray([self._lr_prev_tool], dtype=np.int64)
            self._lr_writer.stream_frame(frame, self._lr_task)

            if self._lr_cam_start_idx is None:
                self._lr_cam_start_idx = dict(cam_indices_cur)

        self._lr_prev = (state_cur, images_cur, cam_indices_cur)
        self._lr_prev_tool = tool_cur
        self._lr_count += 1
        self._lr_next_cap += 1.0 / self._lr_fps

    def save_data(self, episode_video_dir: str | None = None, ep_start_wall: float | None = None, **kwargs) -> None:
        """提交本集数据到后台 worker（不阻塞主线程）。"""
        if self._lr_camera_source == "mp4":
            self._save_data_mp4(episode_video_dir, ep_start_wall)
            return

        if self._lr_count < 2:
            _log.warning(
                f"[LeRobot] 帧数不足（门控次数={self._lr_count}），丢弃本集"
            )
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        self._log_cam_alignment()
        ep_idx = self._lr_writer.flush_episode()
        written = self._lr_count - 1
        _log.info(
            f"[LeRobot] ✓ 提交 {written} 帧（流式落盘），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def _save_data_mp4(self, episode_video_dir: str | None, ep_start_wall: float | None) -> None:
        if episode_video_dir is None:
            _log.error("[LeRobot] mp4 模式 save_data 必须传 episode_video_dir，丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        N = len(self._lr_states)
        if N < 2:
            _log.warning(f"[LeRobot] mp4 模式帧数不足（{N} 条 state 记录），丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        states = [s for s, _ in self._lr_states]
        wall_ts = [t for _, t in self._lr_states]
        ep_start = ep_start_wall if ep_start_wall is not None else (
            self._lr_ep_start_wall if self._lr_ep_start_wall is not None else wall_ts[0]
        )

        _log.info(f"[LeRobot] mp4 模式：从 {episode_video_dir} 逐帧提取 {N} 帧（生成器模式）...")
        cams = camera_keys(self._lr_camera_map)
        frame_gen = iter_frames_from_mp4(
            episode_video_dir, self._lr_camera_map, wall_ts, ep_start, self._lr_target_hw
        )
        for i, images in enumerate(frame_gen):
            if i >= N - 1:
                break
            frame: dict = {
                "observation.state": states[i].astype(np.float32),
                "action": self.build_action(states[i], states[i + 1]).astype(np.float32),
            }
            for cam_key in cams:
                frame[f"observation.images.{cam_key}"] = images[cam_key]
            self._lr_writer.stream_frame(frame, self._lr_task)

        ep_idx = self._lr_writer.flush_episode()
        _log.info(
            f"[LeRobot] ✓ 提交 {N - 1} 帧（MP4 批量提取），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def clear_data(self) -> None:
        """丢弃帧缓存，清理 PNG，重置 episode 状态。"""
        self._lr_writer.discard_episode()
        self._reset_episode()
        try:
            import os
            import shutil
            self.data = {}
            unit = self.get_current_unit_path()
            if os.path.exists(unit):
                shutil.rmtree(unit)
            self.get_next_unit_path()
        except Exception as e:
            _log.warning("[LeRobot] 未能完整清理当前数据单元目录")

    def _reset_episode(self) -> None:
        # 注意 _lr_tool_src 不在这里清 —— 它是配置(整轮采集常驻)，
        # 每集 save_data 都会调本函数，清了第 2 集起就没标签了。
        self._lr_prev: tuple | None = None
        self._lr_prev_tool: int = -1
        self._lr_count: int = 0
        self._lr_next_cap: float | None = None
        self._lr_cam_start_idx: dict | None = None
        self._lr_states: list = []
        self._lr_ep_start_wall: float | None = None

    def _log_cam_alignment(self) -> None:
        if not self._lr_cameras or self._lr_cam_start_idx is None or self._lr_prev is None:
            return
        _, _, cam_end_idx = self._lr_prev
        written = self._lr_count - 1
        if written <= 0:
            return
        for env_name in self._lr_camera_map:
            start = self._lr_cam_start_idx.get(env_name)
            end = cam_end_idx.get(env_name)
            if start is None or end is None:
                continue
            cam_frames = end - start
            ratio = cam_frames / written if written > 0 else 0.0
            if ratio < 0.5:
                _log.warning(
                    f"[LeRobot][相机] {env_name} 更新率偏低（相对采集率 {ratio:.2f}）"
                )
            else:
                _log.info(
                    f"[LeRobot][相机] {env_name} 帧同步正常"
                )


# ---------------------------------------------------------------------------
# G1 OmniPicker 具体子类（18 维 state）
# ---------------------------------------------------------------------------

_G1_OMNIPICKER_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_grip_inner_norm", "l_grip_outer_norm",
    "r_grip_inner_norm", "r_grip_outer_norm",
]


def check_server_state_dim(metadata: dict | None, state_dim: int) -> None:
    """拿策略服务 websocket 握手时给的 metadata 校验 state 维度，不符**直接抛异常**。

    为什么值得为这个加一道硬校验：pi0.5 的 state 是塞进 prompt 的一串数字文字
    （"State: v0 v1 ... vN;"），维度不对**不会**报任何错 —— 模型只是收到一条训练时
    从没出现过的 prompt，分数莫名其妙变差，没有任何征兆。d19 那次评估就是这么白跑掉的。
    服务端 advertise 的 state_dim 来自 TrainConfig.policy_metadata。

    老服务端（没 advertise）只打印一行提示、不拦 —— 那时维度只能靠人眼核对 [state] 那行。
    """
    if not metadata or "state_dim" not in metadata:
        print("[校验] 服务端没 advertise state_dim（旧版服务端，或那档 config 没写 "
              "policy_metadata），跳过校验 —— 此时维度只能靠人眼核对 [state] 那一行。",
              flush=True)
        return
    served = int(metadata["state_dim"])
    if served != state_dim:
        raise RuntimeError(
            f"state 维度不匹配：服务端训练时是 {served} 维，本脚本这一帧会发 {state_dim} 维。"
            f" 服务端 state_names_tail={metadata.get('state_names_tail')}。"
            f" 修法是把 --tool_refs 换成与服务端配套的那份 json"
            f"（d22 模型 → tool_refs_d22.json；d19 模型 → tool_refs_d19goal.json）。"
        )
    print(f"[校验] 服务端 state_dim={served}，与本脚本一致 ✓", flush=True)


# v3「周期感知」新增的 3 维（state 19..21）。名字必须和
# scripts/local/build_d22_cycle_dataset.py 写进 meta/info.json 的 NEW_DIMS 一致。
_CYCLE_STATE_NAMES = ["goal_y", "cycle_dist", "cycle_idx"]
# 闭爪判据，与 build_d22_cycle_dataset.py 的 GRIP_THRESH 一致（取右爪 inner）。
_CYCLE_GRIP_THRESH = 0.5


class G1OmniPickerLeRobotStorage(LeRobotSimSyncMixin, G1OmniPickerDataStorage):
    """G1 OmniPicker 的 LeRobot 格式 storage（18 维 state，可选第 19 维）。

    state (18 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_grip_inner_norm(1), l_grip_outer_norm(1),
         r_grip_inner_norm(1), r_grip_outer_norm(1)]

    第 19 维（可选，只在传入 refs 时启用）：
        到"最近参照点"的欧氏距离（米）。取 min，所以参照点集合怎么定义都行：
          - d19 v1（ref_name="tool_dist"）：参照点 = 5 个工具抓取点。
            抓到手 d≈0，离得远 d 大。放盒那一刻读数是 ~0.27（手在盒里却显示"远"）。
          - d19 v2（ref_name="goal_dist"）：参照点 = 5 个工具抓取点 ∪ 1 个盒子落点。
            抓到手和放进盒都 d≈0，每集归零 10 次而非 5 次。
        算法同一个（下面就是一行 min），只是点集不同 —— 所以换版本不用改代码，
        换 json 即可。**必须与训练用的那份 json 是同一组点**，否则这维是错的。

    第 20~22 维（可选，只在额外传入 cycle_refs 时启用，state 变 22 维）：
        goal_y     —— 本轮"目标工具"的 y 坐标（base 系，米）。
        cycle_dist —— 右末端到那个目标的欧氏距离。
        cycle_idx  —— 轮次号 1..5（正在收第几把），6 = 五把都放完了。
        cycle_refs 是**有序**的 5 个工具抓取点（按抓取顺序），与第 19 维那个
        无序 min 集合是两回事，不能共用一份点。规则必须与
        scripts/local/build_d22_cycle_dataset.py 逐字一致，尤其 idx 只能靠
        **数开爪事件**推进，不能用时间。详见 build_state。

    夹爪归一化：每个 actuator 按各自 actuator_ranges 的 (min, max) 线性映射到 [0, 1]。
    G1 默认 ranges = (-1.0, 2.0)，即 norm = (val + 1.0) / 3.0。

    底盘 /action/drive/ctrl 不写入 LeRobot 数据集（仅用于遥操作时移动机器人）。
    """

    def __init__(
        self,
        dataset_path: str,
        tool_refs: np.ndarray | None = None,
        ref_name: str = "tool_dist",
        cycle_refs: np.ndarray | None = None,
    ) -> None:
        """tool_refs: (N, 3) 参照点（base 系）。名字沿用工具抓取点，含义已泛化为
        "一组参照点"。d19 v2 传的就是 5 工具 + 1 盒子共 6 个点。

        ref_name: 第 19 维在 state_names 里显示的名字，只是元数据，不影响数值。

        cycle_refs: (5, 3) **按抓取顺序**排好的 5 个工具抓取点。传入即启用第
        20~22 维（state 变 22 维），同时 tool_refs 也必须传（第 19 维仍要算）。
        必须来自与训练同一份 tool_refs.json，顺序也要一致。

        默认 None —— 保持 18 维老行为。采集脚本都不传，所以采集出来的数据格式
        一个字节都不变；只有推理路径显式传入，state 才变成 19 或 22 维。
        """
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        from conf import g1_omnipicker_conf
        n_l = len(g1_omnipicker_conf.gripper_l["actuator_names"])
        n_r = len(g1_omnipicker_conf.gripper_r["actuator_names"])
        l_ranges = g1_omnipicker_conf.gripper_l["actuator_ranges"][:n_l]
        r_ranges = g1_omnipicker_conf.gripper_r["actuator_ranges"][:n_r]
        self._l_grip_min = np.array([r[0] for r in l_ranges], dtype=np.float32)
        self._l_grip_max = np.array([r[1] for r in l_ranges], dtype=np.float32)
        self._r_grip_min = np.array([r[0] for r in r_ranges], dtype=np.float32)
        self._r_grip_max = np.array([r[1] for r in r_ranges], dtype=np.float32)
        self._n_l = n_l
        self._n_r = n_r
        self._tool_refs = (
            None if tool_refs is None else np.asarray(tool_refs, dtype=np.float32).reshape(-1, 3)
        )
        self._ref_name = ref_name
        self._cycle_refs = (
            None if cycle_refs is None else np.asarray(cycle_refs, dtype=np.float32).reshape(-1, 3)
        )
        if self._cycle_refs is not None:
            if self._tool_refs is None:
                raise ValueError("cycle_refs 需要同时传 tool_refs（第 19 维仍要算）")
            if self._cycle_refs.shape[0] != 5:
                raise ValueError(
                    f"cycle_refs 必须是 5 个点（按抓取顺序），收到 {self._cycle_refs.shape[0]}"
                )
        self._n_released = 0       # 本试内已"开爪释放"过几次，驱动 cycle_idx
        self._prev_closed = False  # 上一帧右爪是否闭合，用于检测 闭→开 边沿
        # 进度覆盖（2026-09-17 加）：默认 None = 老行为（数开爪边沿）。
        # 设成整数 n 后，cycle_idx 直接取 n+1，不再数边沿。
        # 用途：推理端用"箱子内部范围判定"数出的**真实已入箱件数**来推进轮次。
        # 为什么需要：数开爪边沿的前提是"每次开爪都真的放好了一件"。实测里这个前提
        # 会破 —— 抓空、被碰开、半路掉，都会白记一次，于是 idx 提前跳到下一把，
        # 模型就把还没收的工具**整件跳过**。反过来，模型抓着工具不放（repeat=10
        # 那次抓着螺丝刀在箱子里僵了 128s）时边沿永远不来，idx 卡死在 1，后面
        # 4 把一件都轮不到。两种错都来自"用动作事件代替真实进度"。
        # 训练数据里这两件事是重合的（脚本轨迹是"放到箱底 → 张开"），
        # 所以用箱内容判定既符合训练分布，又只在**出错的回合**里才和训练不同。
        self._progress_override: int | None = None

    @property
    def state_dim(self) -> int:
        if self._cycle_refs is not None:
            return 22
        return 18 if self._tool_refs is None else 19

    @property
    def state_names(self) -> list[str]:
        if self._tool_refs is None:
            return _G1_OMNIPICKER_STATE_NAMES
        if self._cycle_refs is not None:
            return [*_G1_OMNIPICKER_STATE_NAMES, self._ref_name, *_CYCLE_STATE_NAMES]
        return [*_G1_OMNIPICKER_STATE_NAMES, self._ref_name]

    def set_progress_override(self, n_placed: int | None) -> None:
        """把 cycle_idx 的推进源从"数开爪边沿"换成调用方数出来的真实进度。

        n_placed = 已经真的进了箱子的件数（0..5）；传 None 退回老行为。
        调用方必须每帧**在 build_state 之前**调用一次，并且自己保证单调不减
        （箱子判定会因为工具被碰出来而抖动，回退会让模型看到 idx 倒退，
        那是训练分布里从不出现的事）。

        只对 22 维（传了 cycle_refs）有意义；18/19 维模式调用是空操作。
        """
        self._progress_override = None if n_placed is None else int(n_placed)

    def reset_cycle(self) -> None:
        """每一试开始前必须调一次。

        漏调的话上一试的计数会带进来，idx 会从大于 1 开始，模型看到的
        cycle_idx 和训练分布直接错位 —— 这是静默错误，不会报任何异常。
        """
        self._n_released = 0
        self._prev_closed = False
        self._progress_override = None

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 state（18 / 19 / 22 维），夹爪按各自量程归一化到 [0, 1]。

        **有状态**：22 维模式下每调一次都会推进内部的开爪计数器，所以必须
        每帧调且只调一次，别拿它做试验性调用。试与试之间调 reset_cycle()。
        """
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)    # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[:self._n_l]
        r_motor = motor[self._n_l:self._n_l + self._n_r]
        l_range = self._l_grip_max - self._l_grip_min
        r_range = self._r_grip_max - self._r_grip_min
        l_norm = np.clip((l_motor - self._l_grip_min) / np.where(l_range > 0, l_range, 1.0), 0.0, 1.0)
        r_norm = np.clip((r_motor - self._r_grip_min) / np.where(r_range > 0, r_range, 1.0), 0.0, 1.0)
        base = np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ])
        if self._tool_refs is None:
            return base.astype(np.float32)
        # 与训练数据同一套算法：d = min_j || r_pos - ref_j ||。
        # 取 min 意味着参照点集合是"并集"语义，所以 v2 把盒子落点也塞进来即可得到
        # 定义 C，无需改这里。注意必须和训练时用的是同一组点（同一场景布局）。
        d = np.linalg.norm(self._tool_refs - pos[1][None, :], axis=1).min()
        if self._cycle_refs is None:
            return np.concatenate([base, [d]]).astype(np.float32)

        # ---- 第 20~22 维，规则与 build_d22_cycle_dataset.py 逐字对齐 ----
        # idx = 1..5 表示"正在收第几把"，6 表示五把都放完了。
        # **必须靠数开爪事件推进，不能用时间**：仿真快慢一抖，按时间算的轮次
        # 就跟模型的实际进度脱节；而"开爪"是模型自己发出来的动作，天然同步。
        # 顺序要紧：先算 idx（用"截至上一帧"的计数）再更新计数器。训练端是在
        # 开爪帧的**下一帧**才切到下一轮，这样两边逐帧对齐、off-by-one 抵消。
        idx = min(self._n_released + 1, 6)
        # 进度覆盖：调用方（④）用箱子内部范围判定数出真实已入箱件数后设进来。
        # 只影响 idx，不影响 goal_y/cycle_dist 的算法（那两维仍按 idx 查表）。
        if self._progress_override is not None:
            idx = min(max(int(self._progress_override), 0) + 1, 6)
        # idx=6 时目标冻结在最后一把（而不是盒子）—— 与训练数据一致。盒子离
        # 最后一把有 0.65 m，切过去会让这一维瞬间跳变，把归一化尺度带坏。
        target = self._cycle_refs[idx - 1] if idx <= 5 else self._cycle_refs[-1]
        goal_y = float(target[1])
        cycle_dist = float(np.linalg.norm(pos[1] - target))

        # 老行为：数闭→开边沿。覆盖生效时仍然维护计数器（供日志/对照），但不参与 idx。
        closed = bool(r_norm[0] > _CYCLE_GRIP_THRESH)
        if self._prev_closed and not closed:  # 闭 → 开 = 刚释放了一次
            self._n_released += 1
        self._prev_closed = closed

        return np.concatenate(
            [base, [d], [goal_y, cycle_dist, float(idx)]]
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# openloong 具体子类（16 维 state）
# ---------------------------------------------------------------------------

_OPENLOONG_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_gripper",
    "r_gripper",
]


class OpenLoongLeRobotStorage(LeRobotSimSyncMixin, OpenLoongDataStorage):
    """openloong 的 LeRobot 格式 storage（16 维 state）。"""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)
        from conf import openloong_conf
        self._l_grip_max = float(openloong_conf.gripper_l["actuator_ranges"][0][1])
        self._r_grip_max = float(openloong_conf.gripper_r["actuator_ranges"][0][1])

    @property
    def state_dim(self) -> int:
        return 16

    @property
    def state_names(self) -> list[str]:
        return _OPENLOONG_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_grip_norm = float(np.clip(motor[0], 0.0, self._l_grip_max)) / self._l_grip_max
        r_grip_norm = float(np.clip(motor[1], 0.0, self._r_grip_max)) / self._r_grip_max
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            [l_grip_norm, r_grip_norm],
        ]).astype(np.float32)


# ---------------------------------------------------------------------------
# tiangong2 具体子类（38 维 state，24 维灵巧手 effector motor）
# ---------------------------------------------------------------------------

def _build_tiangong2_state_names() -> list[str]:
    names = [
        "l_pos_x", "l_pos_y", "l_pos_z",
        "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
        "r_pos_x", "r_pos_y", "r_pos_z",
        "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    ]
    from conf import tiangong2_conf
    for name in tiangong2_conf.gripper_l["actuator_names"]:
        names.append(f"l_{name}_norm")
    for name in tiangong2_conf.gripper_r["actuator_names"]:
        names.append(f"r_{name}_norm")
    return names


class Tiangong2LeRobotStorage(LeRobotSimSyncMixin, Tiangong2DataStorage):
    """tiangong2 的 LeRobot 格式 storage（灵巧手，38 维 state）。"""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)
        from conf import tiangong2_conf
        n_l = len(tiangong2_conf.gripper_l["actuator_names"])
        n_r = len(tiangong2_conf.gripper_r["actuator_names"])
        self._l_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_l["actuator_ranges"][:n_l]],
            dtype=np.float32,
        )
        self._r_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_r["actuator_ranges"][:n_r]],
            dtype=np.float32,
        )
        self._n_effector = n_l + n_r
        self._n_l = n_l

    @property
    def state_dim(self) -> int:
        return 14 + self._n_effector

    @property
    def state_names(self) -> list[str]:
        return _build_tiangong2_state_names()

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[:self._n_l]
        r_motor = motor[self._n_l:]
        l_norm = np.clip(l_motor, 0.0, self._l_hand_max) / self._l_hand_max
        r_norm = np.clip(r_motor, 0.0, self._r_hand_max) / self._r_hand_max
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ]).astype(np.float32)
