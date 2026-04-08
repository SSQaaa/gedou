import cv2
import apriltag
import time
import logging
from dataclasses import dataclass
from threading import Thread, Lock
from queue import Queue, Empty
from typing import Optional

import config


logger = logging.getLogger(__name__)

TARGET_ID = getattr(config, "APRILTAG_TARGET_ID", 1)
NOT_ID = getattr(config, "APRILTAG_NOT_ID", 2)


@dataclass
class AprilDetection:
    x: int
    y: int
    type: int
    score: float = 1.0


class ApriltagDetector:
    def __init__(self, serial_comm=None):
        self.serial_comm = serial_comm

        self.detector = apriltag.Detector()
        self.queue = Queue(maxsize=1)
        self.running = True

        # 吞吐统计
        self._stat_lock = Lock()
        self._tp_window_start = time.time()
        self._tp_processed = 0

        # 最新结果（主循环读取）
        self._res_lock = Lock()
        self._latest: Optional[AprilDetection] = None
        self._seq: int = 0
        self._ts: float = 0.0

        self.thread = Thread(target=self._detect_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        # 保留最新帧
        if not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                pass
        self.queue.put(frame)

    def get_latest(self):
        """返回 (seq, timestamp, result)。result 可能为 None。"""
        with self._res_lock:
            return int(self._seq), float(self._ts), self._latest

    # ===============================
    # FPS 统计（通俗接口）
    # ===============================
    def _fps_tick(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._stat_lock:
            self._tp_processed += int(n)

    def get_throughput_fps(self, reset: bool = True):
        now = time.time()
        with self._stat_lock:
            elapsed = max(now - self._tp_window_start, 1e-6)
            cnt = int(self._tp_processed)
            fps = float(cnt) / float(elapsed)
            if reset:
                self._tp_window_start = now
                self._tp_processed = 0
            return fps, cnt, float(elapsed)

    def _detect_loop(self):  # apriltag检测
        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)
            except Empty:
                time.sleep(0.005)
                continue

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tags = self.detector.detect(gray)
            except Exception as e:
                logger.exception("[ApriltagDetector] detect exception: %s", e)
                continue

            self._fps_tick(1)

            out: Optional[AprilDetection] = None
            if tags:
                for tag in tags:
                    if tag.center is None:
                        continue
                    x, y = int(tag.center[0]), int(tag.center[1])

                    if tag.tag_id == 0:
                        out = AprilDetection(x=x, y=y, type=1)
                        break
                    elif tag.tag_id == TARGET_ID:
                        out = AprilDetection(x=x, y=y, type=2)
                        break
                    elif tag.tag_id == NOT_ID:
                        out = AprilDetection(x=x, y=y, type=3)
                        break

            now = time.time()
            with self._res_lock:
                self._latest = out
                self._seq += 1
                self._ts = now

    def close(self):
        self.running = False
        self.thread.join()