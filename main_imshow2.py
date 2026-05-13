# -*- coding: utf-8 -*-
"""
离线视频同步检测 + 可视化
- 输入本地视频
- AprilTag 同步检测
- Wheel(RKNN YOLOv5) 同步检测
- 绘制框、typeid、坐标、score
- 输出 mp4 视频
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import apriltag
import numpy as np
import config


# =========================
# 配置
# =========================
VIDEO_PATH = "/home/radxa/fianlcode/videos/777.mp4"
OUTPUT_PATH = "/home/radxa/fianlcode/videos/output_detected2.mp4"

APRILTAG_TARGET_ID = getattr(config, "APRILTAG_TARGET_ID", 1)
APRILTAG_NOT_ID = getattr(config, "APRILTAG_NOT_ID", 2)

IMG_SIZE = config.IMG_SIZE
OBJ_THRESH = config.OBJ_THRESH
NMS_THRESH = config.NMS_THRESH
ANCHORS_PATH = config.ANCHORS_PATH
MODEL_PATH = config.MODEL_PATH
TOPK = getattr(config, "TOPK", 4)
MAX_DIST = getattr(config, "MAX_DIST", 160)
ROI_HEIGHT = float(getattr(config, "ROI_HEIGHT", 1.0))


# =========================
# 读 anchors
# =========================
with open(ANCHORS_PATH, "r") as f:
    values = [float(v) for v in f.readlines()]
    ANCHORS = np.array(values).reshape(3, -1, 2).tolist()


# =========================
# 数据结构
# =========================
@dataclass
class AprilDetection:
    x: int
    y: int
    type: int
    score: float = 1.0
    box: tuple = None
    corners: Optional[np.ndarray] = None
    tag_id: int = -1
    
@dataclass
class WheelDetection:
    x: int
    y: int
    score: float
    type: int = 4
    boxes: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None


# =========================
# 公共函数
# =========================
def roi(frame_bgr, roi_height: float):
    h, w = frame_bgr.shape[:2]
    roi_h = int(h * roi_height)
    roi_h = max(1, min(h, roi_h))
    return frame_bgr[h - roi_h:h, 0:w], h - roi_h


def map_box_with_offset(box, offset_y):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (int(x1), int(y1 + offset_y), int(x2), int(y2 + offset_y))


def put_text_safe(img, text, x, y, color, scale=0.7, thickness=2):
    x = int(max(0, x))
    y = int(max(20, y))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_top_right_info(vis, april_res, wheel_res):
    h, w = vis.shape[:2]
    x0 = w - 380
    y0 = 30
    line_gap = 30

    # 半透明黑底，增强可读性
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0 - 10, 5), (w - 10, 95), (0, 0, 0), -1)
    vis[:] = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    if april_res is not None:
        txt1 = f"AprilTag type={april_res.type} tagid={april_res.tag_id}"
        txt2 = f"AprilTag xy=({int(april_res.x)},{int(april_res.y)})"
    else:
        txt1 = "AprilTag type=None"
        txt2 = "AprilTag xy=(None,None)"

    if wheel_res is not None:
        txt3 = f"Wheel type={wheel_res.type}"
        txt4 = f"Wheel xy=({int(wheel_res.x)},{int(wheel_res.y)})"
    else:
        txt3 = "Wheel type=None"
        txt4 = "Wheel xy=(None,None)"

    put_text_safe(vis, txt1, x0, y0, (0, 0, 255), scale=0.7)
    put_text_safe(vis, txt2, x0, y0 + line_gap, (0, 0, 255), scale=0.7)
    put_text_safe(vis, txt3, x0, y0 + line_gap * 2, (0, 255, 0), scale=0.7)
    put_text_safe(vis, txt4, x0, y0 + line_gap * 3, (0, 255, 0), scale=0.7)


# =========================
# AprilTag 同步检测
# =========================
def detect_apriltag_sync(detector, frame_bgr) -> Optional[AprilDetection]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    if not tags:
        return None

    for tag in tags:
        if tag.center is None or tag.corners is None:
            continue

        x, y = int(tag.center[0]), int(tag.center[1])

        corners = tag.corners
        x1 = int(min(pt[0] for pt in corners))
        y1 = int(min(pt[1] for pt in corners))
        x2 = int(max(pt[0] for pt in corners))
        y2 = int(max(pt[1] for pt in corners))
        box = (x1, y1, x2, y2)

        if tag.tag_id == 0:
            out_type = 1
        elif tag.tag_id == APRILTAG_TARGET_ID:
            out_type = 2
        elif tag.tag_id == APRILTAG_NOT_ID:
            out_type = 3
        else:
            out_type = int(tag.tag_id)

        print("AprilTag detected:",
              "tag_id=", int(tag.tag_id),
              "center=", (x, y),
              "box=", box)

        return AprilDetection(
            x=x,
            y=y,
            type=out_type,
            score=1.0,
            box=box,
            corners=np.array(corners, dtype=np.float32).copy(),
            tag_id=int(tag.tag_id),
        )

    return None


# =========================
# Wheel 同步检测核心
# =========================
def to_nhwc_uint8(rgb_hwc: np.ndarray) -> np.ndarray:
    if rgb_hwc.dtype != np.uint8:
        rgb_hwc = rgb_hwc.astype(np.uint8)
    return np.expand_dims(rgb_hwc, 0)


def letterbox_bgr(img_bgr: np.ndarray, new_shape=(640, 640), color=(0, 0, 0)):
    h, w = img_bgr.shape[:2]
    new_w, new_h = int(new_shape[0]), int(new_shape[1])

    r = min(new_w / w, new_h / h)
    rw, rh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - rw
    pad_h = new_h - rh
    left = pad_w // 2
    top = pad_h // 2

    out = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return out, r, left, top


def unletterbox_boxes_xyxy(boxes_xyxy: np.ndarray, scale: float, pad_left: int, pad_top: int) -> np.ndarray:
    b = boxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] -= float(pad_left)
    b[:, [1, 3]] -= float(pad_top)
    b[:, :4] /= float(scale + 1e-12)
    return b


def center_from_topk_filter(boxes_xyxy: np.ndarray, scores: np.ndarray, topk: int = 4, max_dist_px: float = 160.0):
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


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[pos]
    boxes = boxes[pos]
    classes = classes[pos]
    return boxes, classes, scores


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
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


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


def post_process(input_data, anchors=ANCHORS):
    boxes, scores, classes_conf = [], [], []

    input_data = [
        out.reshape([len(anchors[0]), -1] + list(out.shape[-2:])) for out in input_data
    ]

    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:, :4, :, :], anchors[i]))
        scores.append(input_data[i][:, 4:5, :, :])
        classes_conf.append(input_data[i][:, 5:, :, :])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(b) for b in boxes])
    scores = np.concatenate([sp_flatten(s) for s in scores])
    classes_conf = np.concatenate([sp_flatten(c) for c in classes_conf])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_ = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) > 0:
            nboxes.append(b[keep])
            nclasses.append(c_[keep])
            nscores.append(s[keep])

    if not nclasses:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def detect_wheel_sync(sess, input_name, frame_bgr: np.ndarray) -> Optional[WheelDetection]:
    lb, scale, pad_left, pad_top = letterbox_bgr(
        frame_bgr,
        new_shape=(IMG_SIZE[0], IMG_SIZE[1]),
        color=(0, 0, 0),
    )
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    inp = to_nhwc_uint8(rgb)

    outputs = sess.run(None, {input_name: inp})
    boxes, classes, scores = post_process(outputs, ANCHORS)

    if boxes is None:
        return None

    real_boxes = unletterbox_boxes_xyxy(boxes, scale=scale, pad_left=pad_left, pad_top=pad_top)
    cx, cy, fused_score = center_from_topk_filter(
        real_boxes,
        scores,
        topk=int(TOPK),
        max_dist_px=float(MAX_DIST),
    )

    if fused_score <= 0.0:
        return None

    return WheelDetection(
        x=int(cx),
        y=int(cy),
        score=float(fused_score),
        type=4,
        boxes=real_boxes.copy(),
        scores=scores.copy(),
    )


# =========================
# 绘图
# =========================
def draw_apriltag(vis, det: AprilDetection, roi_offset_y: int):
    if det is None:
        return

    x = int(det.x)
    y = int(det.y + roi_offset_y)

    if det.corners is not None and len(det.corners) == 4:
        pts = det.corners.copy().astype(np.int32)
        pts[:, 1] += int(roi_offset_y)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)

        for p in pts:
            px, py = int(p[0][0]), int(p[0][1])
            cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)

    elif det.box is not None:
        x1, y1, x2, y2 = det.box
        y1 += roi_offset_y
        y2 += roi_offset_y
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)


def draw_wheel(vis, det: WheelDetection, roi_offset_y: int):
    if det is None:
        return

    if det.boxes is not None and det.scores is not None:
        for box, sc in zip(det.boxes, det.scores):
            x1 = int(box[0])
            y1 = int(box[1] + roi_offset_y)
            x2 = int(box[2])
            y2 = int(box[3] + roi_offset_y)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cx = int(det.x)
    cy = int(det.y + roi_offset_y)
    cv2.circle(vis, (cx, cy), 7, (0, 0, 255), -1)


# =========================
# 主函数
# =========================
def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] 视频不存在: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] 无法打开输出视频: {OUTPUT_PATH}")
        cap.release()
        return

    april_detector = apriltag.Detector()

    from ztu_somemodelruntime_ez_rknn_async import InferenceSession, make_provider_options

    provider_options = make_provider_options(layout="nhwc", max_queue_size=1)
    wheel_sess = InferenceSession(
        MODEL_PATH,
        providers=["RknnExecutionProvider"],
        provider_options=provider_options,
    )
    input_name = wheel_sess.input_names[0]

    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_roi = frame
            roi_offset_y = 0
            vis = frame.copy()
            
            april_res = detect_apriltag_sync(april_detector, frame_roi)
            wheel_res = detect_wheel_sync(wheel_sess, input_name, frame_roi)
            
            if april_res is not None:
                draw_apriltag(vis, april_res, 0)
            
            if wheel_res is not None:
                draw_wheel(vis, wheel_res, 0)
            
            draw_top_right_info(vis, april_res, wheel_res)

            writer.write(vis)
            #cv2.imshow("offline_detect", vis)

            #key = cv2.waitKey(1) & 0xFF
            #if key == ord("q"):
            #    break

    finally:
        cap.release()
        writer.release()
        #cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"[INFO] done, output saved to: {OUTPUT_PATH}")
    if elapsed > 1e-6:
        print(f"[INFO] avg_fps={cap.get(cv2.CAP_PROP_FRAME_COUNT) / elapsed:.2f}")


if __name__ == "__main__":
    main()