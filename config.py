# 敌人检测参数
MODEL_PATH = "models/best_n.rknn"
IMG_SIZE = (640, 640)
OBJ_THRESH = 0.4
NMS_THRESH = 0.45
TOPK = 4
MAX_DIST = 160

# wheel detector 配置
IMSHOW_FLAG = False
CLASSES = ("wheel",)
ANCHORS_PATH = "models/anchors_yolov5.txt"

EZ_RKNN_ASYNC_ROOT = "third_party/ztu_somemodelruntime_ez_rknn_async/python"

# 串口参数
SERIAL_PORT = "/dev/ttyS3"
BAUDRATE = 115200
# 串口优先级超时（秒）：超过该时间未收到更高优先级 type，则回退默认态
PRIORITY_TIMEOUT = 0.01

# 摄像头参数
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

ROI_HEIGHT = 0.6