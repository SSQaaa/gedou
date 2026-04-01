import time
from dataclasses import dataclass
from threading import Thread, Lock
from queue import Queue, Empty
from typing import Optional, Tuple

import cv2
import numpy as np

import config


# 从 config 显式取常量，避免未定义
IMSHOW_FLAG = config.IMSHOW_FLAG
IMG_SIZE = config.IMG_SIZE
OBJ_THRESH = config.OBJ_THRESH
NMS_THRESH = config.NMS_THRESH
CLASSES = config.CLASSES
ANCHORS_PATH = config.ANCHORS_PATH
MODEL_PATH = config.MODEL_PATH
TOPK = getattr(config, "TOPK", 4)
MAX_DIST = getattr(config, "MAX_DIST", 160)


with open(ANCHORS_PATH, "r") as f:
    values = [float(v) for v in f.readlines()]
    ANCHORS = np.array(values).reshape(3, -1, 2).tolist()


@dataclass
class WheelDetection:
    x: int
    y: int
    score: float
    type: int = 4


class Wheeldetector:
    """Wheel detector（A 模式）：后台线程推理，主线程提交最新帧并取最新结果。"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        img_size: Tuple[int, int] = IMG_SIZE,
        max_queue_size: int = 3,
        schedule: Optional[int] = None,
    ):
        self.model_path = model_path
        self.img_size = img_size

        # 缓存 ROI 配置（避免 detect 内反复读取 config）
        self.roi_height_ratio = float(getattr(config, "ROI_HEIGHT", 1.0))
        self.roi_height_ratio = max(0.0, min(1.0, self.roi_height_ratio))
        self.roi_enable = self.roi_height_ratio < 0.999999

        # 初始化推理 Session
        from ztu_somemodelruntime_ez_rknn_async import InferenceSession, make_provider_options

        provider_options_kwargs = {
            "layout": "nhwc",
            "max_queue_size": max_queue_size,
        }
        if schedule is not None:
            provider_options_kwargs["schedule"] = schedule

        provider_options = make_provider_options(**provider_options_kwargs)

        self.sess = InferenceSession(
            self.model_path,
            providers=["RknnExecutionProvider"],
            provider_options=provider_options,
        )
        self.input_name = self.sess.input_names[0]

        # 异步：帧队列与结果
        self._frame_q: "Queue[np.ndarray]" = Queue(maxsize=1)
        self._lock = Lock()
        self._latest_result: Optional[WheelDetection] = None
        self._seq: int = 0
        self._ts: float = 0.0

        # 可视化缓存（可选）
        self._vis_lock = Lock()
        self._latest_boxes: Optional[np.ndarray] = None
        self._latest_scores: Optional[np.ndarray] = None
        self._latest_center_xy: Optional[Tuple[int, int]] = None

        # 吞吐 FPS 统计
        self._stat_lock = Lock()
        self._tp_window_start = time.time()
        self._tp_processed = 0

        self._running = False
        self._thread: Optional[Thread] = None

    def get_latest(self):
        """返回 (seq, timestamp, result)。result 可能为 None。"""
        with self._lock:
            return int(self._seq), float(self._ts), self._latest_result

    # ===============================
    # A 模式线程控制
    # ===============================
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        """提交最新帧：队列满则丢旧帧，仅保留最新帧。"""
        if not self._running:
            return
        if not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except Exception:
                pass
        try:
            self._frame_q.put_nowait(frame_bgr)
        except Exception:
            pass

    def get_latest_result(self) -> Optional[WheelDetection]:
        with self._lock:
            return self._latest_result

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

    # ===============================
    # 推理线程
    # ===============================
    def _worker_loop(self) -> None:
        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.1)
            except Empty:
                time.sleep(0.002)
                continue

            try:
                res = self.detect(frame)
            except Exception:
                continue

            now = time.time()
            with self._lock:
                self._latest_result = res
                self._seq += 1
                self._ts = now

            self._fps_tick(1)

    # ===============================
    # 预处理/后处理
    # ===============================
    @staticmethod
    def _to_nhwc_uint8(rgb_hwc: np.ndarray) -> np.ndarray:
        if rgb_hwc.dtype != np.uint8:
            rgb_hwc = rgb_hwc.astype(np.uint8)
        return np.expand_dims(rgb_hwc, 0)

    @staticmethod
    def _letterbox_bgr(img_bgr: np.ndarray, new_shape=(640, 640), color=(0, 0, 0)):
        """简易 letterbox：返回 (resized_img, scale, pad_w, pad_h)。

        先不依赖 py_utils；后续你补齐 coco_utils 后可替换为更完整版本。
        """
        h, w = img_bgr.shape[:2]
        new_w, new_h = int(new_shape[0]), int(new_shape[1])

        r = min(new_w / w, new_h / h)
        rw, rh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)

        pad_w = new_w - rw
        pad_h = new_h - rh
        left = pad_w // 2
        top = pad_h // 2

        out = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=color)
        return out, r, left, top

    @staticmethod
    def _unletterbox_boxes_xyxy(boxes_xyxy: np.ndarray, scale: float, pad_left: int, pad_top: int) -> np.ndarray:
        """把 letterbox 后的 xyxy 映射回原图坐标（近似）。"""
        b = boxes_xyxy.astype(np.float32).copy()
        b[:, [0, 2]] -= float(pad_left)
        b[:, [1, 3]] -= float(pad_top)
        b[:, :4] /= float(scale + 1e-12)
        return b

    @staticmethod
    def _center_from_topk_filter(boxes_xyxy: np.ndarray, scores: np.ndarray, topk: int = 4, max_dist_px: float = 160.0):
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return 0, 0, 0.0

        k = int(min(int(topk), int(len(scores))))
        top_idx = np.argsort(scores)[::-1][:k]
        top_boxes = boxes_xyxy[top_idx]
        top_scores = scores[top_idx]

        cx = (top_boxes[:, 0] + top_boxes[:, 2]) * 0.5
        cy = (top_boxes[:, 1] + top_boxes[:, 3]) * 0.5
        centers = np.stack([cx, cy], axis=1)

        w = top_scores.astype(np.float32)
        w_sum = float(np.sum(w))
        if w_sum <= 1e-6:
            ref = np.mean(centers, axis=0)
        else:
            ref = np.sum(centers * w[:, None], axis=0) / w_sum

        d = np.linalg.norm(centers - ref[None, :], axis=1)
        keep = d <= float(max_dist_px)
        kept_centers = centers[keep]
        kept_scores = top_scores[keep]

        if len(kept_centers) == 0:
            best_idx_local = int(np.argmax(top_scores))
            bx = top_boxes[best_idx_local]
            cx0 = int((float(bx[0]) + float(bx[2])) * 0.5)
            cy0 = int((float(bx[1]) + float(bx[3])) * 0.5)
            return cx0, cy0, float(top_scores[best_idx_local])

        fused = np.mean(kept_centers, axis=0)
        fused_score = float(np.max(kept_scores))
        return int(fused[0]), int(fused[1]), fused_score

    # ===============================
    # 后处理（保留你原先算法）
    # ===============================
    @staticmethod
    def filter_boxes(boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]
        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        return boxes, classes, scores

    @staticmethod
    def nms_boxes(boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
            h1 = np.maximum(0.0, yy2 - yy1 + h[order[1:]] * 0 + 1e-5)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        return np.array(keep)

    @staticmethod
    def box_process(position, anchors):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
        anchors = np.array(anchors).reshape(*np.array(anchors).shape, 1, 1)
        box_xy = position[:, :2, :, :] * 2 - 0.5
        box_wh = (position[:, 2:4, :, :] * 2) ** 2 * anchors
        box_xy += np.concatenate((col, row), axis=1)
        box_xy *= stride
        box = np.concatenate((box_xy, box_wh), axis=1)
        xyxy = np.copy(box)
        xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2
        xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2
        xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2
        xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2
        return xyxy

    @classmethod
    def post_process(cls, input_data, anchors=ANCHORS):
        boxes, scores, classes_conf = [], [], []

        input_data = [
            out.reshape([len(anchors[0]), -1] + list(out.shape[-2:])) for out in input_data
        ]

        for i in range(len(input_data)):
            boxes.append(cls.box_process(input_data[i][:, :4, :, :], anchors[i]))
            scores.append(input_data[i][:, 4:5, :, :])
            classes_conf.append(input_data[i][:, 5:, :, :])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = np.concatenate([sp_flatten(b) for b in boxes])
        scores = np.concatenate([sp_flatten(s) for s in scores])
        classes_conf = np.concatenate([sp_flatten(c) for c in classes_conf])

        boxes, classes, scores = cls.filter_boxes(boxes, scores, classes_conf)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c_ = classes[inds]
            s = scores[inds]
            keep = cls.nms_boxes(b, s)
            if len(keep) > 0:
                nboxes.append(b[keep])
                nclasses.append(c_[keep])
                nscores.append(s[keep])

        if not nclasses:
            return None, None, None
        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

    def detect(self, frame_bgr: np.ndarray) -> Optional[WheelDetection]:
        # 预处理
        lb, scale, pad_left, pad_top = self._letterbox_bgr(
            frame_bgr,
            new_shape=(self.img_size[0], self.img_size[1]),
            color=(0, 0, 0),
        )
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        inp = self._to_nhwc_uint8(rgb)

        outputs = self.sess.run(None, {self.input_name: inp})
        boxes, classes, scores = self.post_process(outputs, ANCHORS)
        if boxes is None:
            with self._vis_lock:
                self._latest_boxes = None
                self._latest_scores = None
                self._latest_center_xy = None
            return None

        # 映射回输入图坐标
        real_boxes = self._unletterbox_boxes_xyxy(boxes, scale=scale, pad_left=pad_left, pad_top=pad_top)

        cx, cy, fused_score = self._center_from_topk_filter(
            real_boxes,
            scores,
            topk=int(TOPK),
            max_dist_px=float(MAX_DIST),
        )

        with self._vis_lock:
            self._latest_boxes = real_boxes.copy() if real_boxes is not None else None
            self._latest_scores = scores.copy() if scores is not None else None
            self._latest_center_xy = (int(cx), int(cy)) if fused_score > 0.0 else None

        if fused_score <= 0.0:
            return None
        return WheelDetection(x=int(cx), y=int(cy), score=float(fused_score), type=4)

    def get_latest_vis(self):
        with self._vis_lock:
            boxes = None if self._latest_boxes is None else self._latest_boxes.copy()
            scores = None if self._latest_scores is None else self._latest_scores.copy()
            center = None if self._latest_center_xy is None else (int(self._latest_center_xy[0]), int(self._latest_center_xy[1]))
            return boxes, scores, center

    def draw_latest_on(self, frame_bgr: np.ndarray) -> np.ndarray:
        boxes, scores, center = self.get_latest_vis()
        if boxes is not None and scores is not None:
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"wheel {float(sc):.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        if center is not None:
            cx, cy = center
            cv2.circle(frame_bgr, (int(cx), int(cy)), 8, (0, 0, 255), thickness=-1)
        return frame_bgr
