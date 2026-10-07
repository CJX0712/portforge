# PortForge · 世界级投资组合优化系统
# 作者：晨星 (CJX0712)
"""错误码体系 E100~E500。每个码对应一类可定位故障，便于 CI 门禁与文档检索。"""

from __future__ import annotations


class PortForgeError(Exception):
    """所有 PortForge 异常的基类。"""

    code = "E000"
    doc = "基类错误"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"[{self.code}] {self.doc}: {message}")


class DataError(PortForgeError):
    code = "E100"
    doc = "数据层错误（非有限值 / 维度不符 / 泄漏）"


class ConfigError(PortForgeError):
    code = "E200"
    doc = "配置错误（ENV 覆盖非法 / schema 校验失败）"


class AllocError(PortForgeError):
    code = "E300"
    doc = "分配器错误（权重未归一 / 含负仓 / 求解失败）"


class OptimError(PortForgeError):
    code = "E400"
    doc = "优化求解错误（QP/LP 不收敛 / 数值奇异）"


class EvalError(PortForgeError):
    code = "E500"
    doc = "评测错误（指标口径不一致 / 回测窗口非法）"


def raise_if(cond: bool, err: PortForgeError) -> None:
    """断言式抛出，便于单测与契约检查。"""
    if cond:
        raise err
