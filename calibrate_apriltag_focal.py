import cv2
import apriltag
import math
import time

import config
from utils.camera import FrameGrabber


WINDOW_NAME = "apriltag focal calibration"


def depth_from_corners(corners, focal_px, tag_size_mm):
    sides = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        sides.append(math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))
    side_px = sum(sides) / max(len(sides), 1)
    if side_px <= 1e-6:
        return 0, 0
    return int(round(float(focal_px) * float(tag_size_mm) / side_px)), side_px


def nothing(_):
    pass


def main():
    tag_size_mm = int(getattr(config, "APRILTAG_SIZE_MM", 50))
    focal_px = int(getattr(config, "APRILTAG_FOCAL_PX", 457))

    cam = FrameGrabber(index=config.CAM_INDEX, width=config.CAM_WIDTH, height=config.CAM_HEIGHT)
    detector = apriltag.Detector()

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("FOCAL_PX", WINDOW_NAME, focal_px, 2000, nothing)

    last_print = 0.0

    try:
        while True:
            frame = cam.get()
            if frame is None:
                time.sleep(0.001)
                continue

            focal_px = cv2.getTrackbarPos("FOCAL_PX", WINDOW_NAME)
            focal_px = max(1, focal_px)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            depth = 0
            side_px = 0
            if tags:
                tag = tags[0]
                if tag.center is not None and tag.corners is not None:
                    depth, side_px = depth_from_corners(tag.corners, focal_px, tag_size_mm)
                    pts = tag.corners.astype(int)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    x, y = int(tag.center[0]), int(tag.center[1])
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"focal={focal_px} depth={depth}mm", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now = time.time()
            if now - last_print >= 0.2:
                print(f"APRILTAG_FOCAL_PX={focal_px} depth={depth}mm side_px={side_px:.1f}")
                last_print = now

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
