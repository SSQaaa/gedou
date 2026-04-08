import time
import logging
import atexit
import subprocess
from threading import Thread, Event

logger = logging.getLogger(__name__)


class BeeperGPIO36:
# 低电平触发
    def __init__(self, enabled: bool, gpio_num: int = 20, beep_ms: int = 100):
        self.enabled = bool(enabled)
        self.gpio_num = int(gpio_num)
        self.beep_ms = int(beep_ms)

        self._trigger = Event()
        self._busy = False

        self.off()
        atexit.register(self.off)

        Thread(target=self._worker, daemon=True).start()
        logger.info("蜂鸣器初始化完成")

    def _gpio(self, val: int):
        subprocess.run(["gpioset", "gpiochip3", f"{self.gpio_num}={val}"], capture_output=True)

    def _worker(self):
        while True:
            self._trigger.wait()
            self._trigger.clear()

            if not self.enabled or self._busy:
                continue
            self._busy = True
            try:
                self._gpio(0)
                time.sleep(self.beep_ms / 1000.0)
            finally:
                self._gpio(1)
                self._busy = False

    def init_high(self):
        self.off()

    def beep_async(self):
        if self.enabled:
            self._trigger.set()

    def off(self):
        try:
            self._gpio(1)
        except Exception:
            pass
