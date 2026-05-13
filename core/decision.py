"""决策（优先级 / 发什么 type）。

目标：把“优先级 + 超时回退”从串口层/各检测器里抽成一个可复用的封装。

约定：type 数值越小优先级越高（与你的 SerialComm.send 保持一致）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DecisionResult:
    x: int
    y: int
    type: int
    score: float = 0.0
    depth: int = 0


class PriorityDecision:
    """优先级决策器：在短时间窗口内锁定更高优先级的 type。

    - type 越小，优先级越高
    - 超过 timeout 未收到更高优先级更新，则回退到默认 type（通常是 5：无目标/默认态）

    用法：
        dec = PriorityDecision(default_type=5, timeout=0.01)
        out = dec.update(x, y, type_val, score)
        # out 即建议发送给下位机的数据
    """

    def __init__(self, default_type: int = 5, timeout: float = 0.01):
        self.default_type = int(default_type)
        self.timeout = float(timeout)

        self._last_type = int(default_type)
        self._last_time = time.time()
        self._last_result = DecisionResult(x=0, y=0, type=int(default_type), depth=0, score=0.0)

    def reset(self) -> None:
        self._last_type = int(self.default_type)
        self._last_time = time.time()
        self._last_result = DecisionResult(x=0, y=0, type=int(self.default_type), depth=0, score=0.0)

    def update(self, x: int, y: int, type_val: int, score: float = 0.0, depth: int = 0) -> DecisionResult:
        now = time.time()

        # 超时回退
        if now - self._last_time > self.timeout:
            self._last_type = int(self.default_type)
            self._last_result = DecisionResult(x=0, y=0, type=int(self.default_type), depth=0, score=0.0)

        type_val = int(type_val)

        # 仅当优先级更高（数值更小）或同优先级时，允许更新
        if type_val <= self._last_type:
            self._last_type = type_val
            self._last_time = now
            self._last_result = DecisionResult(x=int(x), y=int(y), type=type_val, depth=int(depth), score=float(score))
            return self._last_result

        # 不允许抢占，返回当前保持态（仍然发保持的 type；坐标置 0 更安全）
        return self._last_result


# 兼容：提供轻量函数接口（如果你更喜欢函数式调用）
_default_decider: Optional[PriorityDecision] = None


def decide(x: int, y: int, type_val: int, score: float = 0.0, *, timeout: float = 0.01, default_type: int = 5) -> Tuple[int, int, int]:
    """模块级决策入口：返回 (x, y, type)。

    注意：此函数内部维护一个单例决策器，适合单线程主循环。
    """
    global _default_decider
    if _default_decider is None or _default_decider.timeout != float(timeout) or _default_decider.default_type != int(default_type):
        _default_decider = PriorityDecision(default_type=default_type, timeout=timeout)

    r = _default_decider.update(x=x, y=y, type_val=type_val, score=score)
    return r.x, r.y, r.type
