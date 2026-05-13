import cv2
import apriltag
import time
import logging
import math
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
    depth: int = 0


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

    @staticmethod
    def _depth_from_corners(corners) -> int:
        tag_size_mm = float(getattr(config, "APRILTAG_SIZE_MM", 50))
        focal_px = float(getattr(config, "APRILTAG_FOCAL_PX", 457))
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            sides.append(math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))
        side_px = sum(sides) / max(len(sides), 1)
        if side_px <= 1e-6:
            return 0
        return int(round(focal_px * tag_size_mm / side_px))

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
                    depth = self._depth_from_corners(tag.corners) if tag.corners is not None else 0

                    if tag.tag_id == 0:
                        out = AprilDetection(x=x, y=y, type=1, depth=depth)
                        break
                    elif tag.tag_id == TARGET_ID:
                        out = AprilDetection(x=x, y=y, type=2, depth=depth)
                        break
                    elif tag.tag_id == NOT_ID:
                        out = AprilDetection(x=x, y=y, type=3, depth=depth)
                        break

            now = time.time()
            with self._res_lock:
                self._latest = out
                self._seq += 1
                self._ts = now

    def close(self):
        self.running = False
        self.thread.join()
