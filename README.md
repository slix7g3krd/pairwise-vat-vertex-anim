# pairwise-vat-vertex-anim

## 领域

游戏、图形与多媒体

## 主题

顶点动画纹理

## 说明

帧率错配、接缝插值与压缩精度。

初始状态下 `graphics/vat_vertex_anim.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
