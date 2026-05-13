import serial
import logging
import sys
import time
from threading import Lock

import config
from utils.beeper import BeeperGPIO36

logging.basicConfig(
    level=getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")).upper(), logging.INFO),
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
        self.priority_timeout = float(getattr(config, "PRIORITY_TIMEOUT", 0.01))
        self.last_priority_time = time.time()

        # 蜂鸣器（旧代码思路：type==3 时短鸣一次）
        self._beeper = None
        # 由于主循环可能每帧都 send(type=3)，必须做一个很短的节流，
        # 否则 300ms 的短鸣会被高频调用叠加成“持续长鸣”。
        self._beep_min_interval_s = float(getattr(config, "BEEP_MIN_INTERVAL", 0.12))
        self._last_beep_time = 0.0
        try:
            self._beeper = BeeperGPIO36(enabled=bool(getattr(config, "BEEP_FLAG", False)))
            self._beeper.init_high()
        except Exception:
            self._beeper = None

        time.sleep(1)

    def send(self, x, depth, type_val):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_priority_time > self.priority_timeout:
                self.last_type = 5
            if type_val <= self.last_type:
                self.last_priority_time = current_time
                x_hex = max(0, min(0xFFF, int(x)))
                depth_hex = max(0, min(0xFFFF, int(depth)))
                data = f"T{int(type_val):X}X{x_hex:03X}D{depth_hex:04X}\r\n"
                try:
                    self.ser3.write(bytes.fromhex("AABB"))
                    self.ser3.write(data.encode('ascii'))

                    cmd1 = f"SET_NUM(0,{type_val},1);\r\n"
                    cmd2 = f"SET_NUM(1,{x},3);\r\n"
                    cmd3 = f"SET_NUM(2,{depth},3);\r\n"
                    self.ser5.write(cmd1.encode('ascii'))
                    self.ser5.write(cmd2.encode('ascii'))
                    self.ser5.write(cmd3.encode('ascii'))
                except Exception as e:
                    logger.exception("[Serial Send Error] %s", e)
                    return

                # 旧代码思路：type==3 时触发一次短鸣（beep_async 内部会响一小段时间再停）
                if self._beeper and type_val != 5 and bool(getattr(self._beeper, "enabled", True)):
                    if current_time - self._last_beep_time >= self._beep_min_interval_s:
                        self._last_beep_time = current_time
                        self._beeper.beep_async()

                logger.debug("Sent: %s", data.strip())
                self.last_type = type_val

    def close(self):
        try:
            if self._beeper:
                self._beeper.off()
        except Exception:
            pass
        try:
            self.ser3.close()
        except Exception:
            pass
        try:
            self.ser5.close()
        except Exception:
            pass
