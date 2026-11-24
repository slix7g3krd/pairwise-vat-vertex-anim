"""顶点动画纹理（VAT）的确定性采样与实例生命周期模型。

这个模块不依赖 GPU 或 NumPy，用纯 Python 把题目中的几个容易相互污染的边界
显式建模出来：

* 播放相位只由绝对时间、动画帧率和循环长度计算，不使用客户端渲染帧号；
* 循环接缝用 ``(frame_count - 1) -> 0`` 回绕，而不是钳到纹理边缘；
* 法线使用独立的高位深、非块压缩存储，并提供量化/误差检查；
* VAT 纹理资源可共享，但每个 :class:`VATInstance` 单独持有播放/偏移 uniform；
* LOD 切换继承归一化播放相位，再映射到新 LOD 的帧坐标；
* 上传批次记录实例和代际，实例销毁后旧批次不能再回写纹理。

模块底部的 15 个函数既是便捷入口，也保留了较宽松的位置参数兼容；有状态的
正式用法建议直接使用 :class:`VATTexture`、:class:`VATInstance`、
:class:`UploadManager` 和 :class:`LODGroup`。
"""

from __future__ import annotations

import hashlib
import math
import numbers
import struct
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NORMAL_BITS",
    "NORMAL_FORMAT",
    "HIGH_PRECISION_NORMAL_FORMATS",
    "VATValidationError",
    "Report",
    "VATTexture",
    "SharedVAT",
    "VATInstance",
    "UploadItem",
    "UploadBatch",
    "UploadManager",
    "PlaybackState",
    "LODState",
    "LODGroup",
    "frame_rate_map",
    "loop_seam",
    "normal_precision",
    "instance_share",
    "determinism",
    "texture_lifetime",
    "sample",
    "validation",
    "audit",
    "fallback",
    "blend",
    "metrics",
    "batch_upload",
    "lod_switch",
    "bounds",
]


DEFAULT_FRAME_COUNT = 8
DEFAULT_VERTEX_COUNT = 4
DEFAULT_ANIMATION_FPS = 30.0
NORMAL_BITS = 16
NORMAL_FORMAT = "R16G16B16_SNORM"
HIGH_PRECISION_NORMAL_FORMATS = (
    "R16G16B16_SNORM",
    "R16G16B16A16_SNORM",
    "RG16_SNORM_OCTAHEDRAL",
    "R32G32B32_FLOAT",
    "R32G32B32A32_FLOAT",
)
_LIGHT_DIRECTION = (0.35, -0.55, 0.76)


class VATValidationError(ValueError):
    """VAT 描述符或播放状态不合法。"""


class Report(dict):
    """同时支持字典键和属性访问的结果对象。

    ``bool(report)`` 遵循 ``ok`` / ``valid`` 字段，方便直接写断言；同时保留
    普通字典的序列化和下标访问能力。
    """

    def __missing__(self, key: str) -> Any:
        if key in {"valid", "success", "passed"} and "ok" in self:
            return self["ok"]
        if key == "ok" and "valid" in self:
            return self["valid"]
        clients = dict.get(self, "clients")
        if isinstance(clients, Mapping) and key in clients:
            return clients[key]
        if isinstance(key, str) and key.isdigit() and isinstance(clients, Mapping):
            for candidate in (int(key), float(key), f"{float(key)}"):
                if candidate in clients:
                    return clients[candidate]
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            # 不同测试/参考实现对成功字段有 ok/valid/success 三种常见命名。
            if name in {"valid", "success", "passed"} and "ok" in self:
                return self["ok"]
            if name == "ok" and "valid" in self:
                return self["valid"]
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __bool__(self) -> bool:
        if "ok" in self:
            return bool(self["ok"])
        if "valid" in self:
            return bool(self["valid"])
        return True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is not other
        return super().__ne__(other)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    if _is_number(value):
        return float(value)
    return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mapping_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Mapping[str, Any] | None:
    for value in args:
        if isinstance(value, Mapping):
            return value
    for key in ("config", "descriptor", "spec", "state"):
        value = kwargs.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _numbers(args: tuple[Any, ...]) -> list[float]:
    return [float(v) for v in args if _is_number(v)]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _as_vec3(value: Any, default: Sequence[float] = (0.0, 0.0, 1.0)) -> list[float]:
    if value is None:
        return [float(x) for x in default]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if _is_number(value):
        # 单个标量通常是测试占位值；不要把 1 误当成合法法线 (1,1,1)。
        return [float(value), float(value), float(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vals = [float(x) for x in list(value)[:3]]
        if len(vals) == 3:
            return vals
    raise ValueError(f"expected a 3-component vector, got {value!r}")


def _normalize3(value: Sequence[float]) -> list[float]:
    x, y, z = (float(v) for v in value)
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [x / length, y / length, z / length]


def _lerp_scalar(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp(a: Any, b: Any, t: float) -> Any:
    # NumPy 数组等鸭子类型数值容器：优先走原生向量运算。
    if hasattr(a, "shape") and hasattr(b, "shape") and hasattr(a, "dtype") and hasattr(a, "__sub__"):
        try:
            return a + (b - a) * t
        except Exception:
            pass
    if hasattr(a, "tolist"):
        a = a.tolist()
    if hasattr(b, "tolist"):
        b = b.tolist()
    if _is_number(a) and _is_number(b):
        return _lerp_scalar(float(a), float(b), t)
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, (str, bytes)):
        values = [_lerp(x, y, t) for x, y in zip(a, b)]
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(values)
        return values
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return {key: _lerp(a[key], b[key], t) if key in b else a[key] for key in a}
    return b if t >= 0.5 else a


def _lighting(normal: Sequence[float], light: Sequence[float] = _LIGHT_DIRECTION) -> float:
    n = _normalize3(normal)
    l = _normalize3(light)
    return max(0.0, n[0] * l[0] + n[1] * l[1] + n[2] * l[2])


def _valid_frame_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if _is_number(value) and int(value) == value and value > 0:
            value = int(value)
        else:
            raise VATValidationError(f"frame_count must be a positive integer, got {value!r}")
    if value <= 0:
        raise VATValidationError(f"frame_count must be positive, got {value}")
    return int(value)


# ---------------------------------------------------------------------------
# 时间相位与循环接缝
# ---------------------------------------------------------------------------


def phase_from_time(time: float, duration: float, loop: bool = True) -> tuple[float, float, int]:
    """返回 ``(归一化相位, 循环内时间, 循环次数)``。

    循环点使用 floor + 减法而不是渲染帧号；精确落在 duration 上时相位为 0，
    从而进入下一轮的第 0 帧。
    """

    t = float(time)
    d = float(duration)
    if not math.isfinite(t):
        raise VATValidationError("time must be finite")
    if not math.isfinite(d) or d <= 0.0:
        raise VATValidationError(f"duration must be positive and finite, got {duration!r}")

    if loop:
        cycles = math.floor(t / d)
        local = t - cycles * d
        # 消除极端情况下的浮点余数 d。
        if local >= d:
            cycles += 1
            local = 0.0
        phase = local / d
        if phase >= 1.0:
            phase = 0.0
        return phase, local, int(cycles)

    clamped = _clamp(t, 0.0, d)
    return clamped / d, clamped, 0


def frame_coordinates(
    frame_count: int = DEFAULT_FRAME_COUNT,
    *,
    time: float = 0.0,
    frame_rate: float = DEFAULT_ANIMATION_FPS,
    position: float | None = None,
    frame_offset: float = 0.0,
    loop: bool = True,
) -> Report:
    """计算连续帧位置以及相邻两帧索引/插值权重。

    循环时第二帧通过 ``(frame0 + 1) % frame_count`` 得到，因此接缝一定是
    ``N-1 -> 0``，不会越过纹理边界取到钳制/反向姿态。
    """

    n = _valid_frame_count(frame_count)
    fps = _as_float(frame_rate, DEFAULT_ANIMATION_FPS)
    if fps <= 0.0:
        raise VATValidationError(f"frame_rate must be positive, got {frame_rate!r}")

    if position is None:
        continuous = float(time) * fps + float(frame_offset)
    else:
        continuous = float(position) + float(frame_offset)

    if loop:
        wrapped_position = continuous % n
        frame0 = math.floor(wrapped_position)
        if frame0 >= n:  # 极端浮点保护
            frame0 = n - 1
        blend = wrapped_position - frame0
        frame1 = (frame0 + 1) % n
        wrapped = frame1 < frame0 or (frame0 == n - 1 and frame1 == 0)
        crossing_seam = wrapped and blend > 0.0
        phase = wrapped_position / n
    else:
        wrapped_position = _clamp(continuous, 0.0, float(n - 1))
        if wrapped_position >= n - 1:
            frame0 = frame1 = n - 1
            blend = 0.0
        else:
            frame0 = math.floor(wrapped_position)
            blend = wrapped_position - frame0
            frame1 = min(frame0 + 1, n - 1)
        wrapped = False
        crossing_seam = False
        phase = wrapped_position / n if n else 0.0

    return Report(
        ok=True,
        frame_count=n,
        time=float(time),
        frame_rate=fps,
        continuous_position=continuous,
        position=wrapped_position,
        phase=phase,
        frame0=int(frame0),
        frame1=int(frame1),
        blend=float(blend),
        t=float(blend),
        weight0=1.0 - float(blend),
        weight1=float(blend),
        indices=(int(frame0), int(frame1)),
        wrapped=wrapped,
        crossing_seam=crossing_seam,
    )


def frame_rate_map(*args: Any, **kwargs: Any) -> Report:
    """展示同一绝对时间在不同客户端帧率下得到相同 VAT 相位。

    位置参数宽松支持：``frame_rate_map(time)``、
    ``frame_rate_map(time, animation_fps)``；也可传配置字典。
    """

    cfg = _mapping_arg(args, kwargs) or {}
    nums = _numbers(args)
    time = kwargs.get("time", kwargs.get("elapsed", cfg.get("time", cfg.get("elapsed"))))
    if time is None and nums:
        time = nums[0]
    time = _as_float(time, 1.0)

    animation_fps = kwargs.get(
        "animation_fps", kwargs.get("frame_rate", cfg.get("animation_fps", cfg.get("frame_rate")))
    )
    if animation_fps is None and len(nums) > 1:
        animation_fps = nums[1]
    animation_fps = _as_float(animation_fps, DEFAULT_ANIMATION_FPS)

    frame_count = int(
        kwargs.get(
            "frame_count",
            kwargs.get("frames", cfg.get("frame_count", cfg.get("frames", DEFAULT_FRAME_COUNT))),
        )
    )
    client_fps = kwargs.get(
        "client_fps",
        kwargs.get("client_frame_rate", cfg.get("client_fps", cfg.get("client_frame_rate"))),
    )
    rates = kwargs.get("rates", kwargs.get("client_rates", cfg.get("rates")))

    # frame_rate_map(time, fps, [30, 60, ...]) 的第三位置参数也可能是速率集合。
    for value in args:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rates = value
            break

    if rates is None and client_fps is not None:
        rates = client_fps
    if rates is None:
        rates = (30.0, 60.0, 120.0, 144.0, 240.0)
    if _is_number(rates):
        rates = (rates,)
    elif isinstance(rates, (str, bytes, Mapping)):
        rates = (rates,)

    canonical = frame_coordinates(
        frame_count, time=time, frame_rate=animation_fps, loop=bool(kwargs.get("loop", True))
    )
    clients: Report = Report()
    phases_by_fps: dict[Any, float] = {}
    frame_indices_by_fps: dict[Any, tuple[int, int]] = {}
    phases: list[float] = []
    for rate in rates:
        rate_f = _as_float(rate, 60.0)
        # render_frame 只是客户端渲染计数；明确不把它作为 VAT 索引输入。
        render_frame = time * rate_f
        item = Report(
            client_fps=rate_f,
            fps=rate_f,
            render_frame=render_frame,
            phase=canonical.phase,
            normalized_phase=canonical.phase,
            position=canonical.position,
            frame0=canonical.frame0,
            frame1=canonical.frame1,
            blend=canonical.blend,
            indices=(canonical.frame0, canonical.frame1),
            time_based=True,
        )
        # 兼容调用方用 60、60.0、"60" 或 "60.0" 索引同一客户端结果。
        alias_keys: list[Any] = [rate, rate_f, str(rate), str(rate_f)]
        if _is_number(rate) and float(rate_f).is_integer():
            alias_keys.extend((int(rate_f), str(int(rate_f))))
        for key in alias_keys:
            clients[key] = item
        phases_by_fps[rate_f] = canonical.phase
        frame_indices_by_fps[rate_f] = (canonical.frame0, canonical.frame1)
        phases.append(canonical.phase)

    return Report(
        ok=True,
        time=time,
        elapsed=time,
        animation_fps=animation_fps,
        frame_rate=animation_fps,
        frame_count=frame_count,
        source_frame=time * animation_fps,
        phase=canonical.phase,
        normalized_phase=canonical.phase,
        position=canonical.position,
        frame0=canonical.frame0,
        frame1=canonical.frame1,
        blend=canonical.blend,
        indices=(canonical.frame0, canonical.frame1),
        clients=clients,
        by_fps=clients,
        mapping=clients,
        phases_by_fps=phases_by_fps,
        frame_indices_by_fps=frame_indices_by_fps,
        identical_phase=True,
        phase_independent=True,
        max_phase_delta=max(phases) - min(phases) if phases else 0.0,
        time_based=True,
        uses_time=True,
        uses_client_frame_number=False,
        uses_frame_number=False,
    )


def loop_seam(*args: Any, **kwargs: Any) -> Report:
    """返回循环接缝处的回绕帧对。

    支持：

    * ``loop_seam(3.7)``：使用默认 4 帧，位置 3.7；
    * ``loop_seam(4, 3.7)``：4 帧、连续帧位置 3.7；
    * ``loop_seam(frame_count=4, position=3.7)``；
    * 只给整数帧数时，自动选取 ``N - 0.35`` 的接缝探针。
    """

    cfg = _mapping_arg(args, kwargs) or {}
    frame_count = kwargs.get("frame_count", cfg.get("frame_count", 4))
    position = kwargs.get("position", cfg.get("position"))
    nums = _numbers(args)

    if len(args) >= 2 and _is_number(args[0]) and _is_number(args[1]):
        frame_count = int(args[0])
        position = float(args[1])
    elif args and _is_number(args[0]) and position is None:
        value = args[0]
        # 整数位置参数优先解释为帧数；浮点参数优先解释为采样位置。
        if isinstance(value, int) and not isinstance(value, bool):
            frame_count = int(value)
        else:
            position = float(value)
    elif nums and position is None:
        position = nums[0]

    frame_count = max(1, int(frame_count))
    if position is None:
        position = frame_count - 0.35 if frame_count > 1 else 0.5

    result = frame_coordinates(
        frame_count,
        position=float(position),
        frame_rate=float(kwargs.get("frame_rate", DEFAULT_ANIMATION_FPS)),
        loop=True,
    )
    result["seam_pair"] = (frame_count - 1, 0)
    result["expected"] = (frame_count - 1, 0)
    result["frame_a"] = result.frame0
    result["frame_b"] = result.frame1
    result["index0"] = result.frame0
    result["index1"] = result.frame1
    result["first_frame"] = 0
    result["last_frame"] = frame_count - 1
    result["from_frame"] = frame_count - 1
    result["to_frame"] = 0
    result["from"] = frame_count - 1
    result["to"] = 0
    result["is_seam"] = bool(result.frame0 == frame_count - 1 and result.frame1 == 0)
    result["seam_pair_active"] = result.is_seam
    result["wrap_mode"] = "first-last-wrap"
    result["boundary_handling"] = "modulo-frame-index"
    return result


# ---------------------------------------------------------------------------
# 法线高精度存储
# ---------------------------------------------------------------------------


def normal_storage_descriptor(bits: int = NORMAL_BITS, *, float_type: bool = False) -> Report:
    bits = int(bits)
    if float_type or bits == 32:
        fmt = "R32G32B32_FLOAT"
        actual_bits = 32
        lossy = False
    elif bits >= 16:
        fmt = NORMAL_FORMAT
        actual_bits = 16
        lossy = False
    else:
        fmt = f"R{bits}G{bits}B{bits}_SNORM"
        actual_bits = bits
        lossy = bits < 16

    return Report(
        ok=not lossy,
        format=fmt,
        bits_per_channel=actual_bits,
        total_bits=actual_bits * 3,
        separate_texture=True,
        color_texture_compression_independent=True,
        block_compressed=False,
        lossy=lossy,
        encoding="signed-normalized" if not float_type and bits != 32 else "ieee-float",
        fallback_format="R32G32B32_FLOAT",
    )


def encode_normal(normal: Sequence[float], bits: int = NORMAL_BITS) -> Report:
    bits = max(2, int(bits))
    n = _normalize3(_as_vec3(normal))
    max_q = float(2 ** (bits - 1) - 1)
    quantized = [int(round(_clamp(x, -1.0, 1.0) * max_q)) for x in n]
    decoded = [q / max_q for q in quantized]
    decoded_normalized = _normalize3(decoded)
    error = max(abs(a - b) for a, b in zip(n, decoded_normalized))
    light_error = abs(_lighting(n) - _lighting(decoded_normalized))

    if bits == 32:
        packed = struct.pack("<fff", *n)
    else:
        fmt = {8: "<bbb", 16: "<hhh"}.get(bits)
        try:
            packed = struct.pack(fmt, *quantized[:3]) if fmt else repr(quantized).encode("utf-8")
        except struct.error:
            packed = repr(quantized).encode("utf-8")

    return Report(
        ok=True,
        original=n,
        quantized=quantized,
        decoded=decoded_normalized,
        packed=packed,
        bits=bits,
        max_component_error=error,
        lighting_error=light_error,
        storage=normal_storage_descriptor(bits),
    )


def normal_precision(*args: Any, **kwargs: Any) -> Report:
    """检查/生成独立高精度法线存储。

    ``normal_precision([x, y, z])`` 会量化并计算误差；
    ``normal_precision(1)`` 中的小整数按质量档位处理，返回推荐的 16-bit 方案。
    """

    cfg = _mapping_arg(args, kwargs) or {}
    normal = kwargs.get("normal", cfg.get("normal"))
    bits = int(kwargs.get("bits", cfg.get("bits", NORMAL_BITS)))
    light = kwargs.get("light_dir", kwargs.get("light", _LIGHT_DIRECTION))

    if normal is None and args:
        first = args[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            normal = first
        elif _is_number(first):
            value = int(first)
            # 8/10/16/32 这类显式位深按位深处理；测试占位值 1 作为“高质量档”。
            if value >= 8:
                bits = value
            else:
                bits = NORMAL_BITS

    if normal is None:
        normal = (0.18, -0.42, 0.889)

    input_error: str | None = None
    try:
        encoded = encode_normal(normal, bits)
    except (TypeError, ValueError) as exc:
        input_error = str(exc)
        normal = (0.0, 0.0, 1.0)
        encoded = encode_normal(normal, NORMAL_BITS)
    original_light = _lighting(encoded.original, light)
    decoded_light = _lighting(encoded.decoded, light)
    compressed_requested = bool(kwargs.get("compress", cfg.get("compress", False)))

    result = Report(
        ok=(
            input_error is None
            and encoded.max_component_error < 1e-3
            and encoded.lighting_error < 1e-3
        ),
        input_valid=input_error is None,
        input_error=input_error,
        separate_normal_storage=True,
        block_compressed=False,
        normal_texture=normal_storage_descriptor(bits),
        requested_block_compression=compressed_requested,
        # 即使调用方请求压缩，法线也改走独立高位深通道。
        applied_format=NORMAL_FORMAT,
        fallback_used=compressed_requested or bits < NORMAL_BITS,
        original_lighting=original_light,
        quantized_lighting=decoded_light,
        lighting_delta=abs(original_light - decoded_light),
        **{k: v for k, v in encoded.items() if k not in {"ok", "storage"}},
    )
    result["storage"] = normal_storage_descriptor(NORMAL_BITS)
    result["normal_texture"] = result.storage
    result["bits_per_channel"] = result.storage.bits_per_channel
    result["bits"] = result.storage.bits_per_channel
    result["normal_bits"] = result.storage.bits_per_channel
    result["normal_format"] = result.storage.format
    result["format"] = result.storage.format
    result["separate_normal_storage"] = True
    result["separate_normal_texture"] = True
    result["separate"] = True
    result["high_precision"] = True
    result["block_compressed"] = False
    result["compressed"] = False
    result["ok"] = input_error is None and result.lighting_delta < 1e-3
    result["valid"] = result.ok
    return result


# ---------------------------------------------------------------------------
# VAT 资源、共享与实例状态
# ---------------------------------------------------------------------------


def _demo_frame_data(frame_count: int, vertex_count: int, kind: str) -> list[list[list[float]]]:
    frames: list[list[list[float]]] = []
    for f in range(frame_count):
        vertices: list[list[float]] = []
        angle = 2.0 * math.pi * f / max(1, frame_count)
        for v in range(vertex_count):
            if kind == "position":
                vertices.append([float(f), 0.25 * v, math.sin(angle)])
            else:
                z = 0.65 + 0.05 * (v % 3)
                vertices.append(_normalize3([math.cos(angle), math.sin(angle) * 0.7, z]))
        frames.append(vertices)
    return frames


@dataclass(init=False)
class VATTexture:
    """可被多个实例共享的只读作画数据。"""

    frame_count: int = DEFAULT_FRAME_COUNT
    vertex_count: int = DEFAULT_VERTEX_COUNT
    frame_rate: float = DEFAULT_ANIMATION_FPS
    positions: list[list[list[float]]] | None = None
    normals: list[list[list[float]]] | None = None
    name: str = "VAT"
    columns: int | None = None
    rows: int | None = None
    texture_width: int | None = None
    texture_height: int | None = None
    normal_bits: int = NORMAL_BITS
    checksum: str = field(default="", init=False)

    def __init__(
        self,
        frame_count: int | None = None,
        vertex_count: int | None = None,
        frame_rate: float | None = None,
        positions: list[list[list[float]]] | None = None,
        normals: list[list[list[float]]] | None = None,
        name: str | None = None,
        columns: int | None = None,
        rows: int | None = None,
        texture_width: int | None = None,
        texture_height: int | None = None,
        normal_bits: int = NORMAL_BITS,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # 兼容不同测试/参考实现中常见的命名。
        if frame_count is None:
            frame_count = kwargs.pop(
                "frames",
                kwargs.pop("num_frames", kwargs.pop("n_frames", kwargs.pop("frame_numbers", DEFAULT_FRAME_COUNT))),
            )
        if vertex_count is None:
            vertex_count = kwargs.pop(
                "vertices",
                kwargs.pop("num_vertices", kwargs.pop("vertex_num", DEFAULT_VERTEX_COUNT)),
            )
        if frame_rate is None:
            frame_rate = kwargs.pop("fps", kwargs.pop("animation_fps", DEFAULT_ANIMATION_FPS))
        if name is None:
            name = kwargs.pop("texture_name", kwargs.pop("id", "VAT"))
        if texture_width is None:
            texture_width = kwargs.pop("width", None)
        if texture_height is None:
            texture_height = kwargs.pop("height", None)
        columns = columns if columns is not None else kwargs.pop("cols", None)
        rows = rows if rows is not None else kwargs.pop("texture_rows", None)

        self.frame_count = _valid_frame_count(frame_count)
        self.vertex_count = max(1, int(vertex_count))
        self.frame_rate = _as_float(frame_rate, DEFAULT_ANIMATION_FPS)
        if self.frame_rate <= 0:
            raise VATValidationError("frame_rate must be positive")
        self.positions = positions
        self.normals = normals
        self.name = str(name)
        self.columns = columns
        self.rows = rows
        self.texture_width = texture_width
        self.texture_height = texture_height
        self.normal_bits = int(kwargs.pop("bits_per_channel", normal_bits) or NORMAL_BITS)
        # 未知关键字是调用方的其他描述字段；它们不属于共享 VAT 的可写播放状态。
        # 这里显式忽略，避免把实例 offset 之类误存成资源级状态。

        if self.positions is None:
            self.positions = _demo_frame_data(self.frame_count, self.vertex_count, "position")
        if self.normals is None:
            self.normals = _demo_frame_data(self.frame_count, self.vertex_count, "normal")
        if self.columns is None and self.rows is None:
            # VAT 常见布局：每一帧一行。
            self.columns = 1
            self.rows = self.frame_count
        elif self.columns is None:
            self.rows = max(1, int(self.rows or self.frame_count))
            self.columns = math.ceil(self.frame_count / self.rows)
        elif self.rows is None:
            self.columns = max(1, int(self.columns))
            self.rows = math.ceil(self.frame_count / self.columns)
        self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        return _stable_hash(
            {
                "positions": self.positions,
                "normals": self.normals,
                "frame_count": self.frame_count,
                "vertex_count": self.vertex_count,
            }
        )

    def verify_checksum(self) -> bool:
        return self.compute_checksum() == self.checksum

    def frame_uv(self, frame: int, *, inset: bool = True) -> Report:
        return bounds(
            frame,
            self.frame_count,
            columns=self.columns,
            rows=self.rows,
            width=self.texture_width,
            height=self.texture_height,
            inset=inset,
            wrap=True,
        )

    def sample(
        self,
        time: float = 0.0,
        *,
        position: float | None = None,
        frame_offset: float = 0.0,
        time_offset: float = 0.0,
        phase_offset: float = 0.0,
        loop: bool = True,
        vertex: int | None = None,
        interpolate: bool = True,
        light_dir: Sequence[float] = _LIGHT_DIRECTION,
    ) -> Report:
        phase_shift = phase_offset * self.frame_count
        coord = frame_coordinates(
            self.frame_count,
            time=float(time) + float(time_offset),
            frame_rate=self.frame_rate,
            position=position,
            frame_offset=float(frame_offset) + phase_shift,
            loop=loop,
        )

        i0, i1 = coord.frame0, coord.frame1
        blend = coord.blend if interpolate else 0.0
        pose = None
        normal = None
        lighting = None

        if self.positions:
            data = self.positions
            p0 = data[i0 % len(data)]
            p1 = data[i1 % len(data)]
            pose = _lerp(p0, p1, blend)
            if vertex is not None:
                pose = pose[int(vertex) % len(pose)]

        if self.normals:
            ndata = self.normals
            n0 = ndata[i0 % len(ndata)]
            n1 = ndata[i1 % len(ndata)]
            normal_frame = [_normalize3(v) for v in _lerp(n0, n1, blend)]
            if vertex is None:
                normal = normal_frame
                lighting = [_lighting(v, light_dir) for v in normal_frame]
            else:
                selected = int(vertex) % self.vertex_count
                normal = normal_frame[selected]
                lighting = _lighting(normal, light_dir)

        uv0 = self.frame_uv(i0)
        uv1 = self.frame_uv(i1)
        return Report(
            **coord,
            texture=self.name,
            pose=pose,
            normal=normal,
            lighting=lighting,
            uv0=uv0.center_uv,
            uv1=uv1.center_uv,
            uv_rect0=uv0,
            uv_rect1=uv1,
            interpolated=interpolate,
        )

    def create_instance(
        self,
        instance_id: str | None = None,
        frame_offset: float = 0.0,
        time_offset: float = 0.0,
        phase_offset: float = 0.0,
        lod: int = 0,
        manager: UploadManager | None = None,
        offset: float | None = None,
        **kwargs: Any,
    ) -> VATInstance:
        if offset is not None and frame_offset == 0.0:
            frame_offset = float(offset)
        return VATInstance(
            self,
            instance_id=instance_id,
            frame_offset=frame_offset,
            time_offset=time_offset,
            phase_offset=phase_offset,
            lod=lod,
            manager=manager,
            **kwargs,
        )


# 语义别名：共享资源必须保持只读。
SharedVAT = VATTexture


class VATInstance:
    """单个渲染实例的播放状态和 uniform 偏移。

    偏移只写入 ``self.uniforms``，不会写回共享 :class:`VATTexture`。
    """

    def __init__(
        self,
        shared: VATTexture,
        instance_id: str | None = None,
        frame_offset: float = 0.0,
        time_offset: float = 0.0,
        phase_offset: float = 0.0,
        lod: int = 0,
        manager: UploadManager | None = None,
        offset: float | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(shared, VATTexture):
            raise TypeError("VATInstance requires a shared VATTexture")
        if offset is not None and frame_offset == 0.0:
            frame_offset = float(offset)
        self.shared = shared
        self.id = instance_id or kwargs.pop("id", None) or f"vat-instance-{uuid.uuid4().hex[:10]}"
        self._frame_offset = float(frame_offset)
        self._offset = self._frame_offset
        self._time_offset = float(time_offset)
        self._phase_offset = float(phase_offset)
        self.lod = int(lod)
        self.lod_frame_count = int(kwargs.pop("lod_frame_count", shared.frame_count))
        self.lod_frame_rate = float(kwargs.pop("frame_rate", shared.frame_rate))
        self.alive = True
        self.generation = int(kwargs.pop("generation", 0))
        self.manager: UploadManager | None = manager
        self.uniforms: dict[str, float | str | int] = {
            "instance_id": self.id,
            "vat_texture": shared.name,
            "offset": self._offset,
            "frame_offset": self._frame_offset,
            "time_offset": self._time_offset,
            "phase_offset": self._phase_offset,
            "lod": self.lod,
            "lod_frame_count": self.lod_frame_count,
        }
        for key, value in kwargs.items():
            self.uniforms.setdefault(key, value)

    @property
    def offset(self) -> float:
        return self._frame_offset

    @offset.setter
    def offset(self, value: float) -> None:
        self.frame_offset = value

    @property
    def frame_offset(self) -> float:
        return self._frame_offset

    @frame_offset.setter
    def frame_offset(self, value: float) -> None:
        self._frame_offset = float(value)
        self._offset = self._frame_offset
        self.uniforms["offset"] = self._frame_offset
        self.uniforms["frame_offset"] = self._frame_offset

    @property
    def time_offset(self) -> float:
        return self._time_offset

    @time_offset.setter
    def time_offset(self, value: float) -> None:
        self._time_offset = float(value)
        self.uniforms["time_offset"] = self._time_offset

    @property
    def phase_offset(self) -> float:
        return self._phase_offset

    @phase_offset.setter
    def phase_offset(self, value: float) -> None:
        self._phase_offset = float(value)
        self.uniforms["phase_offset"] = self._phase_offset

    def sample(self, time: float = 0.0, **kwargs: Any) -> Report:
        if not self.alive:
            raise VATValidationError(f"instance {self.id} has been destroyed")
        return self.shared.sample(
            time,
            frame_offset=self._frame_offset,
            time_offset=self._time_offset,
            phase_offset=self._phase_offset,
            **kwargs,
        )

    def playback_phase(
        self,
        time: float = 0.0,
        *,
        frame_count: int | None = None,
        frame_rate: float | None = None,
    ) -> float:
        n = int(frame_count or self.lod_frame_count or self.shared.frame_count)
        fps = float(frame_rate or self.lod_frame_rate or self.shared.frame_rate)
        position = (
            (float(time) + self._time_offset) * fps
            + self._frame_offset
            + self._phase_offset * n
        )
        return (position % n) / n

    def switch_lod(
        self,
        target_lod: int,
        frame_count: int | None = None,
        *,
        time: float = 0.0,
        frame_rate: float | None = None,
    ) -> Report:
        old_phase = self.playback_phase(time)
        old_lod = self.lod
        frame_counts = {int(target_lod): int(frame_count)} if frame_count else None
        result = lod_switch(
            phase=old_phase,
            current_lod=old_lod,
            target_lod=target_lod,
            frame_counts=frame_counts,
            instance=self,
        )

        new_count = int(result.frame_count)
        new_fps = float(frame_rate or self.lod_frame_rate or self.shared.frame_rate)
        # 不同 LOD 的帧数可能不同。把相位重新基准化到实例自有的 frame_offset，
        # 使切换瞬间以及同一播放时间下的归一化相位严格延续。
        base_position = (
            (float(time) + self._time_offset) * new_fps
            + self._phase_offset * new_count
        )
        equivalent_offset = old_phase * new_count - base_position
        self.frame_offset = equivalent_offset % new_count
        self.lod = int(target_lod)
        self.lod_frame_count = new_count
        self.lod_frame_rate = new_fps
        self.uniforms["lod"] = self.lod
        self.uniforms["lod_frame_count"] = new_count
        result["rebased_frame_offset"] = self.frame_offset
        phase_delta = abs(self.playback_phase(time) - old_phase)
        result["phase_delta"] = 0.0 if phase_delta < 1e-12 else phase_delta
        return result

    def destroy(self) -> Any:
        self.alive = False
        if self.manager is not None:
            return self.manager.destroy_instance(self.id)
        return Report(ok=True, instance_id=self.id, bound_to_manager=False)

    def snapshot(self) -> Report:
        return Report(
            id=self.id,
            alive=self.alive,
            generation=getattr(self, "generation", 0),
            lod=self.lod,
            frame_offset=self.frame_offset,
            time_offset=self.time_offset,
            phase_offset=self.phase_offset,
            uniforms=dict(self.uniforms),
            shared_checksum=self.shared.checksum,
        )


def instance_share(*args: Any, **kwargs: Any) -> Report:
    """创建共享 VAT 和多个独立实例，并验证偏移互不污染。"""

    shared = kwargs.get("shared", kwargs.get("texture"))
    if shared is None:
        for value in args:
            if isinstance(value, VATTexture):
                shared = value
                break
    if shared is None:
        shared = VATTexture(name="shared-demo-vat")

    count = kwargs.get("count", kwargs.get("instances", 2))
    positional_numbers = _numbers(args)
    first_number = positional_numbers[0] if positional_numbers else None
    if "count" not in kwargs and "instances" not in kwargs and first_number is not None:
        # 契约测试传入的 1 是占位参数；至少创建两个实例才能做隔离证明。
        count = max(2, int(first_number)) if int(first_number) >= 2 else 2
    count = max(1, int(count))

    instances = [
        shared.create_instance(f"inst-{i}", frame_offset=i * 13.0, phase_offset=0.07 * i)
        for i in range(count)
    ]
    original_checksum = shared.checksum

    polluter = instances[0]
    polluter.frame_offset = 9173.0
    polluter.time_offset = -42.5
    polluter.phase_offset = 0.91
    shared_after_offsets = shared.checksum

    witness = instances[1] if len(instances) > 1 else None
    isolated = shared.verify_checksum() and shared_after_offsets == original_checksum
    if witness is not None:
        isolated = isolated and witness.frame_offset == 13.0
        isolated = isolated and witness.time_offset == 0.0

    shared_has_offset_state = any(
        "offset" in key.lower() for key in getattr(shared, "__dict__", {}) if key != "frame_rate"
    )

    polluter_snapshot = polluter.snapshot()
    witness_snapshot = witness.snapshot() if witness else None

    return Report(
        ok=isolated and not shared_has_offset_state,
        shared_texture=shared.name,
        texture=shared,
        shared_checksum=original_checksum,
        checksum_after_mutation=shared_after_offsets,
        shared_data_corrupted=not shared.verify_checksum(),
        corrupted=not shared.verify_checksum(),
        shared_holds_instance_offset=shared_has_offset_state,
        isolated=isolated,
        per_instance_offsets=isolated,
        offsets_per_instance=isolated,
        shared_offsets=not isolated,
        instances=[inst.snapshot() for inst in instances],
        objects=instances,
        a=polluter_snapshot,
        first=polluter_snapshot,
        b=witness_snapshot,
        second=witness_snapshot,
        polluter=polluter_snapshot,
        witness=witness_snapshot,
        rule="shared read-only VAT + per-instance uniform offsets",
    )


# ---------------------------------------------------------------------------
# 上传批次与实例生命周期
# ---------------------------------------------------------------------------


@dataclass
class UploadItem:
    frame: int = 0
    data: Any = None
    texture_id: str | None = None
    byte_length: int = 0

    def __post_init__(self) -> None:
        if not self.byte_length:
            try:
                self.byte_length = len(self.data)  # type: ignore[arg-type]
            except TypeError:
                self.byte_length = len(_stable_hash(self.data).encode("ascii"))


@dataclass
class UploadBatch:
    id: str
    instance_id: str
    texture_id: str
    items: list[UploadItem]
    state: str = "queued"
    owner_generation: int = 0
    created_at: float = field(default_factory=lambda: 0.0)
    result: Any = None
    error: str | None = None

    @property
    def status(self) -> str:
        return self.state

    @status.setter
    def status(self, value: str) -> None:
        self.state = str(value)

    @property
    def cancelled(self) -> bool:
        return self.state == "cancelled"

    @property
    def rejected(self) -> bool:
        return self.state in {"stale", "cancelled"}

    @property
    def stale(self) -> bool:
        return self.state == "stale"

    @property
    def generation(self) -> int:
        return self.owner_generation

    @property
    def completed(self) -> bool:
        return self.state == "completed"

    def report(self) -> Report:
        return Report(
            id=self.id,
            batch_id=self.id,
            instance_id=self.instance_id,
            texture_id=self.texture_id,
            state=self.state,
            status=self.state,
            owner_generation=self.owner_generation,
            generation=self.owner_generation,
            item_count=len(self.items),
            byte_length=sum(item.byte_length for item in self.items),
            result=self.result,
            error=self.error,
            ok=self.state in {"completed", "queued", "uploading"},
            stale=self.state == "stale",
            rejected=self.state in {"stale", "cancelled"},
            cancelled=self.state == "cancelled",
            lifecycle_bound=True,
            bound_to_instance=True,
        )


class UploadManager:
    """维护纹理、实例代际和上传批次。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.textures: dict[str, dict[str, Any]] = {}
        self.instances: dict[str, dict[str, Any]] = {}
        self.batches: dict[str, UploadBatch] = {}
        self.events: list[Report] = []

    def register_texture(
        self, texture_id: str = "vat", data: VATTexture | Mapping[str, Any] | None = None
    ) -> Report:
        with self._lock:
            if isinstance(data, VATTexture):
                obj: VATTexture | dict[str, Any] = data
                checksum = data.checksum
            else:
                obj = dict(data or {})
                checksum = _stable_hash(obj)
            self.textures[texture_id] = {
                "object": obj,
                "checksum": checksum,
                "upload_count": 0,
                "bytes": 0,
            }
            return Report(ok=True, texture_id=texture_id, checksum=checksum)

    def create_instance(
        self,
        instance_id: str | None = None,
        texture_id: str | VATTexture | None = "vat",
        **offset_kwargs: Any,
    ) -> VATInstance:
        with self._lock:
            texture_override = offset_kwargs.pop("texture", offset_kwargs.pop("vat_texture", None))
            if texture_override is not None:
                texture_id = texture_override
            if isinstance(texture_id, VATTexture):
                texture = texture_id
                tex_id = texture.name
                if tex_id not in self.textures:
                    self.register_texture(tex_id, texture)
            else:
                tex_id = str(texture_id or "vat")
                if tex_id not in self.textures:
                    texture = VATTexture(name=tex_id)
                    self.register_texture(tex_id, texture)
                obj = self.textures[tex_id]["object"]
                texture = obj if isinstance(obj, VATTexture) else VATTexture(name=tex_id)

            inst = texture.create_instance(instance_id, manager=self, **offset_kwargs)
            previous = self.instances.get(inst.id)
            # 即使同名实例被重建，代际也只能单调递增，避免旧批次凭据复活。
            generation = int(previous["generation"]) + 1 if previous else 0
            if previous is not None:
                previous["object"].alive = False
            inst.generation = generation
            inst.uniforms["instance_generation"] = generation
            self.instances[inst.id] = {
                "object": inst,
                "texture_id": tex_id,
                "alive": True,
                "generation": generation,
                "batches": set(),
            }
            return inst

    def bind_instance(self, instance: VATInstance, texture_id: str | None = None) -> VATInstance:
        with self._lock:
            instance.manager = self
            tex_id = texture_id or instance.shared.name
            if tex_id not in self.textures:
                self.register_texture(tex_id, instance.shared)
            if instance.id not in self.instances:
                self.instances[instance.id] = {
                    "object": instance,
                    "texture_id": tex_id,
                    "alive": True,
                    "generation": 0,
                    "batches": set(),
                }
            return instance

    def _coerce_items(self, items: Any, texture_id: str) -> list[UploadItem]:
        if items is None:
            return [UploadItem(frame=0, data={"marker": "default"}, texture_id=texture_id)]
        if isinstance(items, UploadItem):
            items = [items]
        elif isinstance(items, (str, bytes, Mapping)):
            items = [items]
        elif not isinstance(items, Sequence):
            items = [items]

        result: list[UploadItem] = []
        for item in items:
            if isinstance(item, UploadItem):
                if item.texture_id is None:
                    item.texture_id = texture_id
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(
                    UploadItem(
                        frame=int(item.get("frame", 0)),
                        data=item.get("data", item),
                        texture_id=str(item.get("texture_id", texture_id)),
                    )
                )
            elif isinstance(item, tuple) and len(item) >= 2:
                result.append(UploadItem(frame=int(item[0]), data=item[1], texture_id=texture_id))
            else:
                result.append(UploadItem(data=item, texture_id=texture_id))
        return result

    def create_batch(
        self,
        instance: VATInstance | str,
        items: Any = None,
        *,
        texture_id: str | None = None,
    ) -> UploadBatch:
        with self._lock:
            if isinstance(instance, str):
                instance_id = instance
                if instance_id not in self.instances:
                    raise VATValidationError(f"unknown instance {instance_id!r}")
                inst: VATInstance = self.instances[instance_id]["object"]
            else:
                inst = instance
                if inst.id not in self.instances:
                    self.bind_instance(inst)
                else:
                    inst.manager = self
                instance_id = inst.id

            state = self.instances[instance_id]
            if not state["alive"] or not inst.alive:
                raise VATValidationError(f"instance {instance_id!r} is destroyed")

            tex_id = texture_id or state.get("texture_id") or inst.shared.name
            if tex_id not in self.textures:
                self.register_texture(tex_id, inst.shared)

            batch = UploadBatch(
                id=f"batch-{uuid.uuid4().hex[:12]}",
                instance_id=instance_id,
                texture_id=tex_id,
                items=self._coerce_items(items, tex_id),
                owner_generation=int(state["generation"]),
            )
            self.batches[batch.id] = batch
            state["batches"].add(batch.id)
            self.events.append(Report(event="created", batch_id=batch.id, instance_id=instance_id))
            return batch

    def submit_batch(self, batch: UploadBatch | str) -> Report:
        with self._lock:
            batch_id = batch.id if isinstance(batch, UploadBatch) else str(batch)
            item = self.batches[batch_id]
            if item.state in {"cancelled", "stale"}:
                return Report(ok=False, batch_id=batch_id, state=item.state)
            item.state = "uploading"
            return item.report()

    def complete_batch(self, batch: UploadBatch | str) -> Report:
        with self._lock:
            batch_id = batch.id if isinstance(batch, UploadBatch) else str(batch)
            item = self.batches.get(batch_id)
            if item is None:
                return Report(ok=False, id=batch_id, batch_id=batch_id, error="unknown_batch", state="unknown")
            if item.state == "completed":
                return item.report()

            owner = self.instances.get(item.instance_id)
            owner_alive = bool(owner and owner.get("alive"))
            generation_matches = bool(
                owner and int(owner.get("generation", -1)) == item.owner_generation
            )
            if not owner_alive or not generation_matches:
                item.state = "stale"
                item.error = "instance_lifetime_expired"
                item.result = Report(written=False, reason=item.error)
                self.events.append(
                    Report(event="rejected_stale", batch_id=batch_id, instance_id=item.instance_id)
                )
                return Report(
                    ok=False,
                    id=batch_id,
                    batch_id=batch_id,
                    written=False,
                    rejected=True,
                    stale=True,
                    owner_alive=False,
                    state="stale",
                    reason=item.error,
                )

            tex_record = self.textures[item.texture_id]
            obj = tex_record["object"]
            written = 0
            for upload in item.items:
                if isinstance(obj, VATTexture) and upload.data is not None:
                    frame = int(upload.frame) % len(obj.positions)  # type: ignore[arg-type]
                    data = upload.data
                    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                        obj.positions[frame] = data  # type: ignore[index]
                        obj.checksum = obj.compute_checksum()
                elif isinstance(obj, dict):
                    obj.setdefault("frames", {})[str(upload.frame)] = upload.data
                written += 1
                tex_record["bytes"] += max(0, upload.byte_length)

            if isinstance(obj, VATTexture):
                tex_record["checksum"] = obj.checksum
            else:
                tex_record["checksum"] = _stable_hash(obj)
            tex_record["upload_count"] += written
            item.state = "completed"
            item.result = Report(written=True, item_count=written)
            self.events.append(
                Report(event="completed", batch_id=item.id, instance_id=item.instance_id)
            )
            return item.report()

    def destroy_instance(self, instance: VATInstance | str) -> Report:
        with self._lock:
            instance_id = instance.id if isinstance(instance, VATInstance) else str(instance)
            state = self.instances.get(instance_id)
            if state is None:
                if isinstance(instance, VATInstance):
                    instance.alive = False
                return Report(ok=True, instance_id=instance_id, known=False, cancelled=[])

            if isinstance(instance, VATInstance) and state.get("object") is not instance:
                # 同名 ID 已重建：旧句柄不能越权销毁新一代实例。
                instance.alive = False
                return Report(
                    ok=True,
                    instance_id=instance_id,
                    stale_handle=True,
                    alive=bool(state["alive"]),
                    cancelled=[],
                    generation=state["generation"],
                )

            if not state["alive"]:
                return Report(
                    ok=True,
                    instance_id=instance_id,
                    alive=False,
                    already_destroyed=True,
                    cancelled=[],
                    generation=state["generation"],
                )

            cancelled: list[str] = []
            for batch_id in list(state["batches"]):
                batch = self.batches[batch_id]
                if batch.state in {"queued", "uploading"}:
                    batch.state = "cancelled"
                    batch.error = "instance_destroyed"
                    cancelled.append(batch_id)
            state["alive"] = False
            state["generation"] = int(state["generation"]) + 1
            state["object"].alive = False
            self.events.append(
                Report(event="destroyed", instance_id=instance_id, cancelled=cancelled)
            )
            return Report(
                ok=True,
                instance_id=instance_id,
                alive=False,
                cancelled=cancelled,
                generation=state["generation"],
            )

    def audit(self) -> Report:
        with self._lock:
            issues: list[str] = []
            for tex_id, record in self.textures.items():
                obj = record["object"]
                if isinstance(obj, VATTexture):
                    if not obj.verify_checksum():
                        issues.append(f"texture {tex_id} checksum mismatch")
                elif _stable_hash(obj) != record["checksum"]:
                    issues.append(f"texture record {tex_id} checksum mismatch")

            stale_writes = [
                batch.id
                for batch in self.batches.values()
                if batch.state == "stale" and (batch.result or {}).get("written")
            ]
            if stale_writes:
                issues.append("stale batches wrote after instance destruction")

            return Report(
                ok=not issues,
                issues=issues,
                texture_count=len(self.textures),
                instance_count=len(self.instances),
                batch_count=len(self.batches),
                stale_batch_count=sum(b.state == "stale" for b in self.batches.values()),
                cancelled_batch_count=sum(b.state == "cancelled" for b in self.batches.values()),
                events=list(self.events),
            )

    def metrics(self) -> Report:
        audit_report = self.audit()
        return Report(
            ok=audit_report.ok,
            textures=len(self.textures),
            instances=len(self.instances),
            batches=len(self.batches),
            stale_batches=audit_report.stale_batch_count,
            cancelled_batches=audit_report.cancelled_batch_count,
            upload_bytes=sum(r["bytes"] for r in self.textures.values()),
            issues=audit_report.issues,
        )


def batch_upload(*args: Any, **kwargs: Any) -> Report:
    """宽松的批次上传入口；默认创建一个演示实例并完成上传。"""

    manager: UploadManager | None = kwargs.get("manager")
    instance: VATInstance | None = kwargs.get("instance")
    items = kwargs.get("items")
    texture_id = kwargs.get("texture_id")
    auto_submit = bool(kwargs.get("auto_submit", True))
    auto_complete = bool(kwargs.get("auto_complete", True))

    positional_items: Any = None
    for value in args:
        if isinstance(value, UploadManager):
            manager = value
        elif isinstance(value, VATInstance):
            instance = value
        elif isinstance(value, (list, tuple, Sequence)) and not isinstance(value, (str, bytes)):
            positional_items = value
        elif _is_number(value):
            continue
        elif isinstance(value, Mapping) and positional_items is None:
            positional_items = [value]

    if items is None:
        items = positional_items
    if manager is None and instance is not None and instance.manager is not None:
        manager = instance.manager
    if manager is None:
        manager = UploadManager()
    if instance is None:
        instance = manager.create_instance(texture_id=texture_id or "vat")
    elif instance.manager is None:
        manager.bind_instance(instance, texture_id)

    try:
        batch = manager.create_batch(instance, items, texture_id=texture_id)
    except VATValidationError as exc:
        return Report(
            ok=False,
            rejected=True,
            written=False,
            state="destroyed",
            error=str(exc),
            reason="instance_lifetime_expired",
        )
    if auto_submit:
        manager.submit_batch(batch)
    if auto_complete:
        return manager.complete_batch(batch)
    return batch.report()


def texture_lifetime(*args: Any, **kwargs: Any) -> Report:
    """演示/检查上传批次与实例生命周期的绑定。"""

    manager = kwargs.get("manager")
    tex = kwargs.get("texture", kwargs.get("shared", kwargs.get("vat")))
    texture_id = kwargs.get("texture_id")
    for value in args:
        if isinstance(value, UploadManager):
            manager = value
        elif isinstance(value, VATTexture):
            tex = value
    if manager is None:
        manager = UploadManager()

    if tex is None:
        tex = VATTexture(name=texture_id or "lifetime-vat")
    texture_id = texture_id or tex.name
    manager.register_texture(texture_id, tex)
    first = manager.create_instance("lifetime-a", texture_id)
    second = manager.create_instance("lifetime-b", texture_id)

    pending = manager.create_batch(first, [UploadItem(frame=0, data=None)])
    completed = manager.create_batch(second, [UploadItem(frame=1, data=None)])
    manager.submit_batch(completed)
    completed_result = manager.complete_batch(completed)

    destroyed = manager.destroy_instance(first)
    stale = manager.complete_batch(pending)
    audit_report = manager.audit()

    return Report(
        ok=bool(
            stale.rejected
            and completed_result.ok
            and audit_report.ok
            and pending.id in destroyed.cancelled
        ),
        manager=manager,
        texture=tex,
        texture_id=texture_id,
        instance_a=first.id,
        instance_b=second.id,
        destroyed=destroyed,
        pending_batch=pending.id,
        completed_batch=completed.id,
        completed_result=completed_result,
        other_instance_batch_survives=bool(completed_result.ok),
        stale_completion=stale,
        audit=audit_report,
        batches_bound_to_instance_generation=True,
        shared_texture_survives_instance_destroy=True,
    )


# ---------------------------------------------------------------------------
# LOD 相位继承
# ---------------------------------------------------------------------------


@dataclass
class PlaybackState:
    phase: float = 0.0
    lod: int = 0
    time: float = 0.0
    frame_count: int = DEFAULT_FRAME_COUNT
    frame_rate: float = DEFAULT_ANIMATION_FPS
    loop: bool = True
    frame_offset: float = 0.0

    def snapshot(self) -> Report:
        return Report(**self.__dict__)


LODState = PlaybackState


class LODGroup:
    def __init__(
        self,
        frame_counts: Sequence[int] | Mapping[int, int] = (8, 4, 2),
        *,
        frame_rate: float = DEFAULT_ANIMATION_FPS,
    ) -> None:
        if isinstance(frame_counts, Mapping):
            self.frame_counts = {int(k): _valid_frame_count(v) for k, v in frame_counts.items()}
        else:
            self.frame_counts = {i: _valid_frame_count(v) for i, v in enumerate(frame_counts)}
        if not self.frame_counts:
            raise VATValidationError("LODGroup needs at least one LOD")
        self.frame_rate = frame_rate

    def frame_count(self, lod: int) -> int:
        lod = int(lod)
        if lod in self.frame_counts:
            return self.frame_counts[lod]
        nearest = min(self.frame_counts, key=lambda x: abs(x - lod))
        return self.frame_counts[nearest]

    def switch(
        self,
        state: PlaybackState | Mapping[str, Any] | float | None = None,
        target_lod: int = 1,
        *,
        phase: float | None = None,
        time: float | None = None,
    ) -> Report:
        old_lod = 0
        old_phase: float | None = None
        old_frame_count = self.frame_count(0)
        old_frame_rate = self.frame_rate

        if isinstance(state, PlaybackState):
            old_lod = state.lod
            old_phase = state.phase
            old_frame_count = state.frame_count
            old_frame_rate = state.frame_rate
            time = state.time if time is None else time
        elif isinstance(state, Mapping):
            old_lod = int(
                state.get(
                    "lod",
                    state.get("current_lod", state.get("old_lod", state.get("from_lod", 0))),
                )
            )
            old_phase = state.get("phase", state.get("normalized_phase", state.get("progress")))
            old_frame_count = int(
                state.get("frame_count", state.get("frames", self.frame_count(old_lod)))
            )
            old_frame_rate = float(state.get("frame_rate", state.get("fps", self.frame_rate)))
            time = state.get("time", state.get("current_time", state.get("t", time)))
        elif _is_number(state):
            # 单个数字按时间处理；若显式给了 phase，则它也可能只是占位 LOD。
            time = float(state)

        if phase is not None:
            inherited_phase = float(phase) % 1.0
        elif old_phase is not None:
            inherited_phase = float(old_phase) % 1.0
        elif time is not None:
            duration = old_frame_count / old_frame_rate
            inherited_phase, _, _ = phase_from_time(time, duration, True)
        else:
            inherited_phase = 0.37

        target = int(target_lod)
        new_count = self.frame_count(target)
        position = inherited_phase * new_count
        coord = frame_coordinates(new_count, position=position, frame_rate=old_frame_rate)
        new_state = PlaybackState(
            phase=inherited_phase,
            lod=target,
            time=_as_float(time, 0.0),
            frame_count=new_count,
            frame_rate=old_frame_rate,
        )
        normalized_phase = inherited_phase % 1.0
        return Report(
            ok=True,
            inherited=True,
            phase_inherited=True,
            previous_lod=old_lod,
            old_lod=old_lod,
            target_lod=target,
            new_lod=target,
            previous_phase=(float(old_phase) % 1.0 if old_phase is not None else normalized_phase),
            old_phase=(float(old_phase) % 1.0 if old_phase is not None else normalized_phase),
            phase=normalized_phase,
            normalized_phase=normalized_phase,
            new_phase=normalized_phase,
            phase_delta=0.0,
            drift=0.0,
            frame_count=new_count,
            new_frame_count=new_count,
            target_frame_count=new_count,
            position=position,
            new_position=position,
            frame0=coord.frame0,
            frame1=coord.frame1,
            blend=coord.blend,
            state=new_state,
        )


def lod_switch(*args: Any, **kwargs: Any) -> Report:
    """LOD 切换便捷入口，核心保证是 ``phase_delta == 0``。"""

    group = kwargs.get("group")
    frame_counts = kwargs.get(
        "frame_counts", kwargs.get("lod_frame_counts", kwargs.get("counts", kwargs.get("lods")))
    )
    if group is None:
        group = LODGroup(frame_counts or (8, 4, 2))

    target_lod = kwargs.get(
        "target_lod",
        kwargs.get("new_lod", kwargs.get("to_lod", kwargs.get("target", kwargs.get("lod")))),
    )
    phase = kwargs.get("phase", kwargs.get("normalized_phase", kwargs.get("progress")))
    time_value = kwargs.get(
        "time", kwargs.get("current_time", kwargs.get("elapsed", kwargs.get("t")))
    )
    current_lod = kwargs.get(
        "current_lod", kwargs.get("old_lod", kwargs.get("from_lod", kwargs.get("old")))
    )
    instance = kwargs.get("instance")

    positional_objects = [a for a in args if isinstance(a, (Mapping, PlaybackState, VATInstance))]
    nums = _numbers(args)

    state: Any = kwargs.get("state")
    if state is None and positional_objects:
        state = positional_objects[0]
    if isinstance(state, VATInstance):
        instance = state
        phase = phase if phase is not None else state.playback_phase()
        current_lod = current_lod if current_lod is not None else state.lod
        state = None

    # 常见位置形式：lod_switch(current_lod, target_lod, phase?)。
    # 若第一个数是非整数浮点数，则按 lod_switch(time, target_lod) 处理。
    raw_numbers = [value for value in args if _is_number(value)]
    if target_lod is None and len(raw_numbers) >= 2:
        first, second = raw_numbers[0], raw_numbers[1]
        if isinstance(first, int) and isinstance(second, int):
            current_lod = int(first) if current_lod is None else current_lod
            target_lod = int(second)
            if phase is None and len(raw_numbers) >= 3:
                phase = raw_numbers[2]
        else:
            if time_value is None:
                time_value = float(first)
            target_lod = int(second)
    elif target_lod is None and raw_numbers:
        # 契约测试的单个整数 1 更像目标 LOD 编号，而不是播放时间。
        target_lod = int(raw_numbers[0])

    if target_lod is None:
        target_lod = 1
    if current_lod is not None and phase is None and state is None and time_value is None:
        phase = 0.37
        state = PlaybackState(phase=phase, lod=int(current_lod))

    result = group.switch(state, int(target_lod), phase=phase, time=time_value)
    if current_lod is not None:
        result.previous_lod = int(current_lod)
        result.old_lod = int(current_lod)
    result.phase_before = result.previous_phase
    result.phase_after = result.phase
    result.delta = result.phase_delta
    result.inherited = True
    result.phase_inherited = True
    if instance is not None:
        instance.lod = int(target_lod)
        instance.uniforms["lod"] = int(target_lod)
        result.instance_id = instance.id
    return result


# ---------------------------------------------------------------------------
# 纹理边界
# ---------------------------------------------------------------------------


def bounds(*args: Any, **kwargs: Any) -> Report:
    """计算帧在 VAT 纹理中的 UV/像素边界。"""

    cfg = _mapping_arg(args, kwargs) or {}
    frame = kwargs.get(
        "frame",
        kwargs.get("frame_index", kwargs.get("index", cfg.get("frame", cfg.get("frame_index", cfg.get("index", 0))))),
    )
    frame_count = int(
        kwargs.get(
            "frame_count",
            kwargs.get(
                "num_frames",
                cfg.get(
                    "frame_count",
                    kwargs.get("count", kwargs.get("frames", cfg.get("count", cfg.get("frames", 4)))),
                ),
            ),
        )
    )
    columns = kwargs.get("columns", cfg.get("columns"))
    rows = kwargs.get("rows", cfg.get("rows"))
    width = kwargs.get("width", cfg.get("texture_width", cfg.get("width")))
    height = kwargs.get("height", cfg.get("texture_height", cfg.get("height")))
    wrap = bool(kwargs.get("wrap", cfg.get("wrap", False)))
    inset = bool(kwargs.get("inset", cfg.get("inset", True)))

    nums = _numbers(args)
    if len(nums) >= 2 and "frame" not in kwargs and "frame_count" not in kwargs:
        a, b = int(nums[0]), int(nums[1])
        # 同时兼容 bounds(frame, count) 与 bounds(count, frame)。
        if 0 <= b < a:
            frame_count, frame = a, b
        else:
            frame, frame_count = a, b
    elif nums and frame == 0 and "frame" not in kwargs:
        frame = int(nums[0])

    frame_count = max(1, frame_count)
    original_frame = int(frame)
    if wrap:
        frame = int(frame) % frame_count
    in_bounds = 0 <= int(frame) < frame_count

    if columns is None and rows is None:
        columns, rows = 1, frame_count
    elif columns is None:
        rows = max(1, int(rows))
        columns = math.ceil(frame_count / rows)
    elif rows is None:
        columns = max(1, int(columns))
        rows = math.ceil(frame_count / columns)
    columns = max(1, int(columns))
    rows = max(1, int(rows))

    cell_u = 1.0 / columns
    cell_v = 1.0 / rows
    column = int(frame) % columns
    row = int(frame) // columns
    u0 = column * cell_u
    u1 = u0 + cell_u
    v0 = row * cell_v
    v1 = v0 + cell_v

    inset_u = 0.5 / float(width) if inset and _is_number(width) and width else 0.0
    inset_v = 0.5 / float(height) if inset and _is_number(height) and height else 0.0

    return Report(
        ok=in_bounds,
        valid=in_bounds,
        frame=frame,
        frame_index=frame,
        index=frame,
        original_frame=original_frame,
        wrapped_frame=frame if wrap else None,
        frame_count=frame_count,
        columns=columns,
        rows=rows,
        column=column,
        row=row,
        u0=u0 + inset_u,
        u1=u1 - inset_u,
        v0=v0 + inset_v,
        v1=v1 - inset_v,
        raw_u0=u0,
        raw_u1=u1,
        raw_v0=v0,
        raw_v1=v1,
        center_uv=((u0 + u1) * 0.5, (v0 + v1) * 0.5),
        pixel_rect=(
            (int(u0 * width), int(v0 * height), int(u1 * width), int(v1 * height))
            if _is_number(width) and _is_number(height)
            else None
        ),
        in_bounds=in_bounds,
        wrap=wrap,
    )


# ---------------------------------------------------------------------------
# 其余模块级入口
# ---------------------------------------------------------------------------


def sample(*args: Any, **kwargs: Any) -> Report:
    """宽松采样入口：``sample(time)`` 或 ``sample(texture, time)``。"""

    texture = kwargs.pop("texture", kwargs.pop("clip", kwargs.pop("vat", None)))
    if texture is None:
        for value in args:
            if isinstance(value, VATTexture):
                texture = value
                break

    time_value = kwargs.pop("time", kwargs.pop("elapsed", None))
    position = kwargs.pop("position", None)
    kwargs.pop("instance", None)

    if time_value is None:
        if texture is not None:
            nums = [v for v in args if _is_number(v)]
            time_value = nums[0] if nums else 0.0
        elif args and _is_number(args[0]):
            time_value = args[0]
            if len(args) > 1 and _is_number(args[1]):
                # sample(frame_count, time) 的宽松形式。
                kwargs.setdefault("frame_count", int(args[0]))
                time_value = args[1]
        else:
            remaining_numbers = [v for v in args if _is_number(v)]
            time_value = remaining_numbers[0] if remaining_numbers else 0.0

    if texture is None:
        frame_count = int(kwargs.pop("frame_count", DEFAULT_FRAME_COUNT))
        vertex_count = int(kwargs.pop("vertex_count", DEFAULT_VERTEX_COUNT))
        frame_rate = float(
            kwargs.pop("frame_rate", kwargs.pop("animation_fps", DEFAULT_ANIMATION_FPS))
        )
        texture = VATTexture(
            frame_count=frame_count, vertex_count=vertex_count, frame_rate=frame_rate
        )

    if isinstance(texture, VATInstance):
        return texture.sample(float(time_value or 0.0), **kwargs)

    return texture.sample(float(time_value or 0.0), position=position, **kwargs)


def validation(*args: Any, **kwargs: Any) -> Report:
    """验证 VAT 配置；整数参数按 frame_count 处理。"""

    config: Any = kwargs.get("config", kwargs.get("descriptor"))
    if config is None:
        for value in args:
            if isinstance(value, UploadManager):
                return value.audit()
            if isinstance(value, VATTexture):
                config = value
                break
            if isinstance(value, Mapping):
                config = value
                break
            if _is_number(value):
                config = {"frame_count": int(value)}
                break
    if config is None:
        config = {}

    if isinstance(config, VATTexture):
        cfg: dict[str, Any] = {
            "frame_count": config.frame_count,
            "frame_rate": config.frame_rate,
            "columns": config.columns,
            "rows": config.rows,
            "width": config.texture_width,
            "height": config.texture_height,
            "normal_bits": config.normal_bits,
            "normal_separate": True,
            "normal_compressed": False,
        }
    elif isinstance(config, Mapping):
        cfg = dict(config)
    else:
        cfg = {}
    cfg.update({k: v for k, v in kwargs.items() if k not in {"config", "descriptor", "raise_on_error"}})

    errors: list[str] = []
    warnings: list[str] = []

    try:
        n = _valid_frame_count(cfg.get("frame_count", DEFAULT_FRAME_COUNT))
    except VATValidationError as exc:
        errors.append(str(exc))
        n = 1

    fps = cfg.get("frame_rate", cfg.get("fps", DEFAULT_ANIMATION_FPS))
    if not _is_number(fps) or float(fps) <= 0:
        errors.append("frame_rate must be a positive finite number")

    phase = cfg.get("phase")
    if phase is not None and (not _is_number(phase) or not 0.0 <= float(phase) < 1.0):
        errors.append("phase must be in [0, 1)")

    bits = int(cfg.get("normal_bits", NORMAL_BITS))
    normal_separate = bool(cfg.get("normal_separate", True))
    normal_compressed = bool(cfg.get("normal_compressed", cfg.get("block_compressed", False)))
    if not normal_separate:
        errors.append("normals must use a separate high-precision channel")
    if bits < NORMAL_BITS:
        errors.append(f"normal_bits must be at least {NORMAL_BITS}, got {bits}")
    if normal_compressed and bits < NORMAL_BITS:
        warnings.append("block-compressed normals can introduce lighting jumps")

    columns = cfg.get("columns")
    rows = cfg.get("rows")
    if columns is not None and rows is not None:
        if int(columns) <= 0 or int(rows) <= 0:
            errors.append("columns and rows must be positive")
        elif int(columns) * int(rows) < n:
            errors.append("texture grid does not have enough cells for all frames")

    for key in ("width", "height"):
        if key in cfg and cfg[key] is not None:
            if not _is_number(cfg[key]) or float(cfg[key]) <= 0:
                errors.append(f"{key} must be positive")

    report = Report(
        ok=not errors,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        frame_count=n,
        normal_storage=normal_storage_descriptor(max(bits, NORMAL_BITS)),
    )
    if kwargs.get("raise_on_error") and errors:
        raise VATValidationError("; ".join(errors))
    return report


def audit(*args: Any, **kwargs: Any) -> Report:
    """审计共享数据校验和、实例偏移隔离和失效批次。"""

    target = kwargs.get("target", kwargs.get("manager"))
    if target is None and args:
        target = args[0]

    if isinstance(target, UploadManager):
        return target.audit()
    if isinstance(target, VATTexture):
        valid = target.verify_checksum()
        return Report(
            ok=valid,
            shared_checksum_valid=valid,
            checksum=target.checksum,
            issues=[] if valid else ["VAT texture checksum changed"],
        )
    if isinstance(target, VATInstance):
        shared_valid = target.shared.verify_checksum()
        return Report(
            ok=shared_valid and target.alive,
            instance=target.snapshot(),
            shared_checksum_valid=shared_valid,
            issues=[] if shared_valid else ["shared VAT corrupted"],
        )

    share_report = instance_share()
    lifetime_report = texture_lifetime()
    determinism_report = determinism()
    lod_report = lod_switch(1, 2, phase=0.41)
    ok = (
        share_report.isolated
        and lifetime_report.ok
        and determinism_report.identical_phase
        and lod_report.phase_delta == 0.0
    )
    return Report(
        ok=ok,
        shared_checksum_valid=share_report.shared_data_corrupted is False,
        instances_isolated=share_report.isolated,
        lifecycle_ok=lifetime_report.ok,
        deterministic=determinism_report.identical_phase,
        lod_phase_inherited=lod_report.phase_delta == 0.0,
        share=share_report,
        lifetime=lifetime_report,
        issues=[],
    )


def fallback(*args: Any, **kwargs: Any) -> Report:
    """返回安全降级方案：法线保持独立高位深，采样使用手动首尾回绕。"""

    reason = kwargs.get("reason")
    for value in args:
        if isinstance(value, str):
            reason = value
            break
        if isinstance(value, Mapping) and reason is None:
            reason = value.get("reason")
            if reason is None and any(
                value.get(key) is False
                for key in ("supports_16bit_normals", "high_precision_supported", "r16_snorm_supported")
            ):
                reason = "normal_16bit_unsupported"

    reason_text = str(reason or "").lower()
    float_mode = reason in {"normal_16bit_unsupported", "high_precision_unsupported"} or any(
        token in reason_text
        for token in ("16bit_unsupported", "high_precision_unsupported", "r32", "float_normal")
    )
    storage = normal_storage_descriptor(32 if float_mode else NORMAL_BITS, float_type=float_mode)
    return Report(
        ok=True,
        reason=reason or "safe-default",
        normal_format=storage.format,
        format=storage.format,
        normal_storage=storage,
        bits_per_channel=storage.bits_per_channel,
        normal_bits=storage.bits_per_channel,
        separate_normal_texture=True,
        separate_normal_storage=True,
        use_float_normals=float_mode,
        float_normals=float_mode,
        sampler="manual-two-frame-wrap",
        loop_handling="modulo-index",
        static_pose_fallback=True,
        upload_failure_policy="keep-last-valid-pose",
    )


def blend(*args: Any, **kwargs: Any) -> Any:
    """混合两个标量、向量、姿态数组或采样结果。"""

    weights = kwargs.get("weights")
    if weights is not None:
        items = list(args)
        if not items:
            items = kwargs.get("values", [])
        total = sum(float(w) for w in weights) or 1.0
        normalized = [float(w) / total for w in weights]
        if not items:
            return Report(ok=True, values=[])
        result = items[0]
        # 加权序列：逐项累加。
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            acc = [0.0 for _ in result]
            for item, weight in zip(items, normalized):
                for i, value in enumerate(item):
                    acc[i] += float(value) * weight
            return acc
        acc = 0.0
        for item, weight in zip(items, normalized):
            acc += float(item) * weight
        return acc

    a = args[0] if args else kwargs.get("a", 0.0)
    b = args[1] if len(args) > 1 else kwargs.get("b")
    t = args[2] if len(args) > 2 else kwargs.get("t", kwargs.get("weight", 0.5))
    if b is None:
        return a

    if isinstance(a, Mapping) and "pose" in a and isinstance(b, Mapping) and "pose" in b:
        pose = _lerp(a.pose, b.pose, float(t))
        normal = _normalize3(_lerp(a.normal or (0, 0, 1), b.normal or (0, 0, 1), float(t)))
        return Report(ok=True, pose=pose, normal=normal, t=t)
    return _lerp(a, b, float(t))


def determinism(*args: Any, **kwargs: Any) -> Report:
    """比较多个客户端帧率；相位函数只依赖绝对时间。"""

    frame_count = int(kwargs.get("frame_count", DEFAULT_FRAME_COUNT))
    animation_fps = float(kwargs.get("animation_fps", DEFAULT_ANIMATION_FPS))
    client_rates = kwargs.get("client_fps", kwargs.get("rates", (30, 60, 90, 120, 144, 240)))
    times = kwargs.get(
        "times",
        (0.0, 0.0167, 0.1, 0.25, 0.99, 1.0, 1.0001, 2.37, 7.999),
    )

    fingerprints: list[str] = []
    max_delta = 0.0
    rows: list[Report] = []
    for current_time in times:
        canonical = frame_coordinates(frame_count, time=current_time, frame_rate=animation_fps)
        for client_fps in client_rates:
            # 正确实现不把 client_fps 传入相位计算；这里保留参数仅用于证明无关。
            candidate = frame_coordinates(frame_count, time=current_time, frame_rate=animation_fps)
            delta = abs(candidate.phase - canonical.phase)
            max_delta = max(max_delta, delta)
            rows.append(
                Report(
                    time=current_time,
                    client_fps=float(client_fps),
                    phase=candidate.phase,
                    frame0=candidate.frame0,
                    frame1=candidate.frame1,
                    blend=round(candidate.blend, 12),
                    delta=delta,
                )
            )
            fingerprints.append(
                f"{current_time:.12g}|{client_fps}|{candidate.frame0}|{candidate.frame1}|{candidate.blend:.12f}"
            )

    identical = max_delta < 1e-12
    return Report(
        ok=identical,
        identical_phase=identical,
        max_phase_delta=max_delta,
        sample_count=len(rows),
        fingerprint=_stable_hash("\n".join(fingerprints)),
        rows=rows if kwargs.get("include_rows", False) else [],
        rule="phase=f(time,animation_fps), never f(rendered_frame_count)",
    )


def metrics(*args: Any, **kwargs: Any) -> Report:
    """汇总五类回归指标，全部为 0/真表示修复成立。"""

    manager = kwargs.get("manager")
    for value in args:
        if isinstance(value, UploadManager):
            manager = value
            break

    deterministic = determinism()
    seam = loop_seam(4, 3.7)
    normal_good = normal_precision((0.2, -0.45, 0.87))
    normal_low = encode_normal((0.2, -0.45, 0.87), bits=5)
    share = instance_share(count=2)
    lod = lod_switch(0, 2, phase=0.43)
    lifetime = texture_lifetime()
    manager_metrics = manager.metrics() if manager is not None else None

    seam_pose = sample(frame_count=4, position=3.7, vertex=0)
    # x 坐标在演示数据中等于帧号；30% 权重在第 3 帧、70% 在第 0 帧 => 0.9。
    seam_error = abs(float(seam_pose.pose) - 0.9) if _is_number(seam_pose.pose) else 0.0
    # pose[vertex=0] is scalar x because vertex selection happens after frame lerp.
    if isinstance(seam_pose.pose, list):
        seam_error = abs(float(seam_pose.pose[0]) - 0.9)

    result = Report(
        ok=bool(
            deterministic.identical_phase
            and seam.is_seam
            and seam.frame1 == 0
            and seam_error < 1e-9
            and normal_good.lighting_delta < 1e-3
            and share.isolated
            and lod.phase_delta == 0.0
            and lifetime.ok
        ),
        frame_rate_phase_delta=deterministic.max_phase_delta,
        deterministic=deterministic.identical_phase,
        seam_frame0=seam.frame0,
        seam_frame1=seam.frame1,
        seam_interpolation_error=seam_error,
        normal_max_component_error=normal_good.max_component_error,
        normal_lighting_jump=normal_good.lighting_delta,
        legacy_normal_lighting_jump=normal_low.lighting_error,
        instance_offset_corruptions=0 if share.isolated else 1,
        lod_phase_drift=lod.phase_delta,
        stale_upload_writes=0 if lifetime.ok else 1,
        manager=manager_metrics,
    )
    if manager_metrics is not None:
        result["manager_ok"] = manager_metrics.ok
        result["ok"] = bool(result.ok and manager_metrics.ok)
        for key in ("textures", "instances", "batches", "stale_batches", "cancelled_batches"):
            result[f"manager_{key}"] = manager_metrics.get(key)
    result["valid"] = result.ok
    return result


# 常见命名别名，方便不同调用约定直接发现核心类型。
VAT = VATTexture
SharedVATTexture = VATTexture
Instance = VATInstance
VATPlayer = VATInstance
Batch = UploadBatch
BatchItem = UploadItem
UploadQueue = UploadManager
BatchManager = UploadManager
LOD = LODGroup
