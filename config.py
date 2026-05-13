MYCOLOR = "YELLOW"  # "BLUE"
IMSHOW_FLAG = 1

# 日志等级："DEBUG" / "INFO" / "WARNING" / "ERROR"
LOG_LEVEL = "INFO"

# 蜂鸣器开关
BEEP_FLAG = True

APRILTAG_TARGET_ID = 1 if MYCOLOR == "YELLOW" else 2
APRILTAG_NOT_ID = 2 if MYCOLOR == "YELLOW" else 1


# 敌人检测配置
MODEL_PATH = "models/best_n.rknn"
ANCHORS_PATH = "models/anchors_yolov5.txt"
EZ_RKNN_ASYNC_ROOT = "third_party/ztu_somemodelruntime_ez_rknn_async/python"
CLASSES = ("wheel",)
IMG_SIZE = (640, 640)
OBJ_THRESH = 0.6
NMS_THRESH = 0.45
TOPK = 4
MAX_DIST = 160

# 串口参数
SERIAL_PORT = "/dev/ttyS3"
BAUDRATE = 115200
# 串口优先级超时（秒）：超过该时间未收到更高优先级 type，则回退默认态
PRIORITY_TIMEOUT = 0.01

# 摄像头参数
CAM_INDEX = "/dev/video0"
CAM_WIDTH = 1280
CAM_HEIGHT = 1024

ROI_HEIGHT = 0.7

APRILTAG_SIZE_MM = 50
APRILTAG_FOCAL_PX = 457
