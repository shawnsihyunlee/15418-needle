import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    return rand(
        fan_in,
        fan_out,
        low=-gain * math.sqrt(6.0 / (fan_in + fan_out)),
        high=gain * math.sqrt(6.0 / (fan_in + fan_out)),
        **kwargs,
    )

def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    return randn(
        fan_in,
        fan_out,
        std=gain * math.sqrt(2.0 / (fan_in + fan_out)),
        **kwargs,
    )


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(6.0 / fan_in)
    if shape is not None:
        return rand(
            *shape,
            low=-bound,
            high=bound,
            **kwargs,
        )
    return rand(
        fan_in,
        fan_out,
        low=-bound,
        high=bound,
        **kwargs,
    )


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    return randn(
        fan_in,
        fan_out,
        std=math.sqrt(2.0 / fan_in),
        **kwargs,
    )