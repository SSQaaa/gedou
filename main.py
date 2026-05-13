# -*- coding: utf-8 -*-
import time
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
    return frame_bgr[h - roi_h : h, 0:w]


def main():
    serial_comm = SerialComm(port=config.SERIAL_PORT, baudrate=config.BAUDRATE)

    apriltag_det = ApriltagDetector(serial_comm=None)

    wheel_det = Wheeldetector(model_path=config.MODEL_PATH, img_size=config.IMG_SIZE)
    wheel_det.start()

    # type 数值越小优先级越高
    decider = PriorityDecision(default_type=5, timeout=0.01)

    cam = FrameGrabber(index=config.CAM_INDEX, width=config.CAM_WIDTH, height=config.CAM_HEIGHT)

    roi_height_ratio = float(getattr(config, "ROI_HEIGHT", 1.0))

    last_print = time.time()

    last_april_seq = 0
    last_wheel_seq = 0

    last_wheel_res = None

    try:
        while True:
            frame = cam.get()
            if frame is None:
                time.sleep(0.001)
                continue

            frame_roi = roi(frame, roi_height_ratio)

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
                    out = decider.update(april_res.x, april_res.y, april_res.type, score=april_res.score, depth=april_res.depth)

                if last_wheel_res is not None:
                    out = decider.update(last_wheel_res.x, last_wheel_res.y, last_wheel_res.type, score=last_wheel_res.score)

                if out is None:
                    serial_comm.send(0, 0, 5)
                else:
                    serial_comm.send(out.x, out.depth, out.type)

            now = time.time()
            if now - last_print >= 1.0:
                a_fps, a_cnt, a_elapsed = apriltag_det.get_throughput_fps(reset=True)
                w_fps, w_cnt, w_elapsed = wheel_det.get_throughput_fps(reset=True)
                print(f"[FPS] apriltag={a_fps:.1f} ({a_cnt}/{a_elapsed:.2f}s) | wheel={w_fps:.1f} ({w_cnt}/{w_elapsed:.2f}s)")
                last_print = now

            if config.IMSHOW_FLAG:
                vis = wheel_det.draw_latest_on(frame_roi.copy())
                cv2.imshow("main", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
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
