# -*- coding: utf-8 -*-
import time
import os
import cv2
import config
from core.decision import PriorityDecision
from detectors.apriltag_detector import ApriltagDetector
from detectors.wheel_detector import Wheeldetector
from utils.camera import FrameGrabber
from utils.serial_comm import SerialComm


def roi(frame_bgr, roi_height: float):
    h, w = frame_bgr.shape[:2]
    roi_h = int(h * roi_height)
    roi_h = max(1, min(h, roi_h))
    return frame_bgr[h - roi_h: h, 0:w]


def create_video_writer(frame_width, frame_height, fps=30, save_dir="videos"):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(save_dir, f"record_{timestamp}.mp4")

    # mp4 常用写法，某些板子不支持时可换成 XVID + .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        raise RuntimeError(f"faild:{video_path}")

    print(f"[INFO]start record {video_path}")
    return writer, video_path


def main():
    serial_comm = SerialComm(port=config.SERIAL_PORT, baudrate=config.BAUDRATE)

    apriltag_det = ApriltagDetector(serial_comm=None)

    wheel_det = Wheeldetector(model_path=config.MODEL_PATH, img_size=config.IMG_SIZE)
    wheel_det.start()

    # type 数值越小优先级越高
    decider = PriorityDecision(default_type=5, timeout=0.01)

    cam = FrameGrabber(
        index=config.CAM_INDEX,
        width=config.CAM_WIDTH,
        height=config.CAM_HEIGHT
    )

    roi_height_ratio = float(getattr(config, "ROI_HEIGHT", 1.0))

    last_print = time.time()

    last_april_seq = 0
    last_wheel_seq = 0

    last_wheel_res = None

    video_writer = None
    video_path = None

    try:
        # 先等到第一帧，拿到尺寸后初始化录像器
        while True:
            first_frame = cam.get()
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                # 这里 fps 你可以改，比如 30 / 60
                video_writer, video_path = create_video_writer(w, int(h*roi_height_ratio), fps=30, save_dir="videos")
                break
            time.sleep(0.001)

        while True:
            frame = cam.get()
            if frame is None:
                time.sleep(0.001)
                continue
                
            frame_roi = roi(frame, roi_height_ratio)

            # ===== 保存原始画面 =====
            if video_writer is not None:
                video_writer.write(frame_roi)

            apriltag_det.update_frame(frame_roi)
            wheel_det.submit_frame(frame_roi)

            wheel_seq, _wheel_ts, wheel_res = wheel_det.get_latest()
            if wheel_seq != last_wheel_seq:
                last_wheel_seq = wheel_seq
                last_wheel_res = wheel_res

            april_seq, _april_ts, april_res = apriltag_det.get_latest()
            if april_seq != last_april_seq:
                last_april_seq = april_seq
                out = None
                if april_res is not None:
                    out = decider.update(april_res.x, april_res.y, april_res.type, score=april_res.score)

                if last_wheel_res is not None:
                    out = decider.update(last_wheel_res.x, last_wheel_res.y, last_wheel_res.type, score=last_wheel_res.score)

                if out is None:
                    serial_comm.send(0, 0, 5)
                else:
                    serial_comm.send(out.x, out.y, out.type)

            now = time.time()
            if now - last_print >= 1.0:
                a_fps, a_cnt, a_elapsed = apriltag_det.get_throughput_fps(reset=True)
                w_fps, w_cnt, w_elapsed = wheel_det.get_throughput_fps(reset=True)
                # print(f"[FPS] apriltag={a_fps:.1f} ({a_cnt}/{a_elapsed:.2f}s) | wheel={w_fps:.1f} ({w_cnt}/{w_elapsed:.2f}s)")
                last_print = now

            if config.IMSHOW_FLAG:
                vis = wheel_det.draw_latest_on(frame_roi.copy())
                cv2.imshow("main", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        try:
            if video_writer is not None:
                video_writer.release()
                print(f"[INFO]save to{video_path}")
        except Exception:
            pass
        try:
            wheel_det.stop()
        except Exception:
            pass
        try:
            apriltag_det.close()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        try:
            serial_comm.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()