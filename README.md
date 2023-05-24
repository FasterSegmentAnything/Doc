目录
- [说明](#说明)
- [实验环境](#实验环境)
- [关于](#关于)

# 说明
分析[segment-anything](https://segment-anything.com/)的工作原理、部署流程、使用技巧及其它。

# 实验环境

> 无特殊指定的情况下，本文档默认指代环境如下：

硬件：
| 类型 | 简述 |
| :-- | :-- |
| 名称 | NVIDIA出品</br>Jetson Orin Nano Developer Kit(8GB) |
|  CPU | `6`核`ARM 64`位`Cortex-A78AE v8.2` |
|  GPU | `Ampere`架构</br>`1024 NVIDIA CUDA核心` + `32 Tensor核心` | 
| 内存/显存 | 共用</br>`8GB` |
| 存储 | `128GB M.2 SSD</br>无SD卡` |
| 功率 | `7W` ~ `15W` |
| `AI`算力 | `40TOPS` |
| 参考价格（RMB）| 3700 |


软件：

| 类型 | 简述 |
| :-- | :-- |
| 系统 | `Ubuntu 18.04` |


# 关于

* [论文原文](files\related\paper.pdf)
* [官方网站](https://segment-anything.com/)
* 作者
  * yutian(https://www.aflyingfish.top/)