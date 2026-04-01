import serial
import logging
import sys
import time
from threading import Lock

import config

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SerialComm:
    def __init__(self, port=None, baudrate=None):
        if port is None:
            port = getattr(config, "SERIAL_PORT", "/dev/ttyS3")
        if baudrate is None:
            baudrate = getattr(config, "BAUDRATE", 115200)

        try:
            self.ser3 = serial.Serial(port, baudrate, timeout=0.5)
            self.ser5 = serial.Serial('/dev/ttyS5', 115200, timeout=1)
        except Exception as e:
            logger.exception("[Serial Init Error] %s", e)
            sys.exit(1)

        self.lock = Lock()
        self.last_type = 5
        # 统一从 config 读取（若未配置则用原默认值）
        self.priority_timeout = float(getattr(config, "PRIORITY_TIMEOUT", 0.01))
        self.last_priority_time = time.time()
        time.sleep(1)

    def send(self, x, y, type_val):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_priority_time > self.priority_timeout:
                self.last_type = 5
            if type_val <= self.last_type:
                self.last_priority_time = current_time
                data = f"&x={int(x)}&y={int(y)}&type={type_val}&\r\n"
                try:
                    self.ser3.write(bytes.fromhex("AABB"))
                    self.ser3.write(data.encode('ascii'))
                    # self.ser3.flush()

                    cmd1 = f"SET_NUM(0,{type_val},1);\r\n"
                    cmd2 = f"SET_NUM(1,{x},3);\r\n"
                    cmd3 = f"SET_NUM(2,{y},3);\r\n"
                    self.ser5.write(cmd1.encode('ascii'))
                    self.ser5.write(cmd2.encode('ascii'))
                    self.ser5.write(cmd3.encode('ascii'))
                    # self.ser5.flush()
                except Exception as e:
                    logger.exception("[Serial Send Error] %s", e)
                    return
                logger.debug("Sent: %s", data.strip())
                self.last_type = type_val

    def close(self):
        try:
            self.ser3.close()
        except Exception:
            pass
        try:
            self.ser5.close()
        except Exception:
            pass