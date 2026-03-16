下面是一份 **详细项目归档版**，适合你后续直接保存到笔记、项目文档或仓库 `README` 里。

---

# RK3588 / 鲁班猫 YOLO11-seg → RKNN → 板端缺陷检测

## 详细项目归档文档

## 1. 项目概述

本项目目标是将训练好的 **YOLO11n-seg 印刷缺陷检测模型** 部署到 **RK3588（鲁班猫开发板）**，完成板端图片推理。

本次项目最终实现的是：

* 在 PC / Codespaces 侧完成 `ONNX → RKNN`
* 在 RK3588 板端加载 `.rknn`
* 批量读取图片文件夹
* **忽略 segmentation 分支**
* **只输出 defect 检测框**
* 保存可视化结果图

最终闭环流程如下：

```text
best.pt → best.onnx → best.rknn → RK3588板端推理 → 结果图输出
```

---

## 2. 任务背景

### 2.1 模型任务性质

本项目原始模型为：

```text
YOLO11n-seg
```

训练任务本质上属于：

```text
detect + segmentation
```

也就是说，模型同时具备：

* 检测框输出
* 分割 mask 输出

但当前业务阶段只要求：

```text
先完成 defect 检测
```

所以部署策略是：

* **模型保留 seg 结构**
* **板端推理时忽略 mask 分支**
* **仅做检测框后处理**

---

## 3. 模型信息

### 3.1 模型基本参数

* 模型名称：`best.pt`
* 模型类型：`YOLO11n-seg`
* 类别数：`1`
* 类别名：`defect`
* 输入尺寸：`640 × 640`
* 任务场景：印刷缺陷检测

### 3.2 数据特点

训练数据包含：

* 正常样本
* 缺陷样本

部署阶段的测试图片同样来自实际印刷场景，文件名较长且不规则，因此后续板端脚本采用：

```text
自动遍历 images 文件夹
```

而不是写死单张图片文件名。

---

## 4. 最终跑通的环境与版本

---

### 4.1 PC / Codespaces 侧环境

用于完成：

* ONNX 检查
* RKNN 转换
* 量化构建

最终稳定版本组合如下：

#### 系统环境

* 平台：GitHub Codespaces
* 系统：Ubuntu 24.04

#### Python 版本

* `Python 3.11.9`

#### 关键 Python 包

* `rknn-toolkit2==2.3.2`
* `onnx==1.18.0`
* `onnxruntime==1.18.0`
* `numpy==1.26.4`

#### 系统依赖

* `libgl1`
* `libglib2.0-0`
* `libsm6`
* `libxext6`
* `libxrender1`

---

### 4.2 板端环境

用于完成：

* 加载 `.rknn`
* 初始化 runtime
* 批量图片推理
* 输出可视化结果

#### 板端硬件

* 开发板：鲁班猫
* 芯片平台：RK3588

#### 板端系统 Python

* `Python 3.9.2`

#### 板端 Python 包

* `rknn-toolkit-lite2==1.6.0`
* `numpy==2.0.2`
* `opencv-python-headless==4.13.0.92`

#### 板端 Runtime

* `librknnrt.so version: 1.6.0`
* `Driver version: 0.9.8`

---

## 5. 项目目录结构

---

### 5.1 Codespaces 侧目录结构

```text
rknn_convert/
├── best.onnx
├── best.onnx.data
├── best.rknn
├── dataset.txt
├── dataset/
│   ├── *.png
├── convert.py
├── check0_base_optimize.onnx
├── check2_correct_ops.onnx
└── check3_fuse_ops.onnx
```

### 5.2 各文件说明

* `best.onnx`：导出的 ONNX 网络结构
* `best.onnx.data`：ONNX 外部权重文件
* `dataset/`：INT8 量化图片
* `dataset.txt`：量化数据列表
* `convert.py`：ONNX → RKNN 转换脚本
* `best.rknn`：最终生成的 RKNN 模型
* `check*.onnx`：RKNN 构建时产生的中间优化文件

说明：

`check*.onnx` 文件仅用于转换调试，不需要部署到板端。

---

### 5.3 开发板侧目录结构

```text
/home/cat/
├── best.rknn
├── images/
│   ├── *.png
├── results/
├── check_best_rknn_folder.py
└── infer_defect_folder.py
```

### 5.4 板端各文件说明

* `best.rknn`：部署模型
* `images/`：待推理图片目录
* `results/`：结果输出目录
* `check_best_rknn_folder.py`：最小验证脚本
* `infer_defect_folder.py`：正式批量检测脚本

---

## 6. 完整实施流程

---

### 6.1 第一步：PT → ONNX

在前置阶段，先使用 YOLO11 导出 ONNX，得到：

```text
best.onnx
best.onnx.data
```

然后对 ONNX 进行校验，确认：

* 模型可正常加载
* 输入 shape 正确
* ONNX 文件未损坏

本项目导出的输入节点为：

```text
images
shape = [1, 3, 640, 640]
```

说明模型导出时采用的是：

```text
NCHW
```

布局。

---

### 6.2 第二步：准备 INT8 量化数据集

RKNN 转换时启用了量化，因此需要准备量化图片。

量化目录结构如下：

```text
dataset/
    *.png
dataset.txt
```

`dataset.txt` 中每行放一张图片路径，例如：

```text
dataset/A_blackspot_xxx.png
dataset/A_friction_xxx.png
dataset/B_colorshift_xxx.png
```

本项目量化集规模：

* 约 150 张图片

量化集来源：

* 实际印刷缺陷业务图片

量化集特点：

* 包含正常图
* 包含缺陷图

---

### 6.3 第三步：Codespaces 环境重建

最开始尝试过在 Python 3.12 环境下继续推进，但遇到了 ONNX 安装和 RKNN 兼容问题，最终放弃。

最后采用的稳定方案是：

```text
Python 3.11.9 + ONNX 1.18.0 + RKNN Toolkit2 2.3.2
```

这是本项目最终实际跑通的转换环境。

---

### 6.4 第四步：转换过程中遇到的核心问题

#### 问题一：APT 更新失败

在 Codespaces 中执行 `sudo apt update` 时，曾被 `yarn` 源的 GPG 公钥问题阻塞。

根因：

* 第三方 Yarn 源签名失效或未配置公钥

解决方式：

* 删除 `/etc/apt/sources.list.d/yarn.list`
* 然后重新执行 `sudo apt update`

---

#### 问题二：`rknn-toolkit2` 导入失败

初次导入 `from rknn.api import RKNN` 时出现：

```text
ImportError: libGL.so.1: cannot open shared object file
```

根因：

* Codespaces 缺少 OpenCV 运行依赖

解决方式：

安装：

* `libgl1`
* `libglib2.0-0`

必要时补充：

* `libsm6`
* `libxext6`
* `libxrender1`

之后 `RKNN import OK`。

---

#### 问题三：ONNX 与 RKNN 接口兼容问题

在执行 `rknn.load_onnx()` 时，先后遇到：

```text
AttributeError: module 'onnx' has no attribute 'mapping'
```

以及：

```text
AttributeError: module 'onnx._mapping' has no attribute 'TENSOR_TYPE_TO_NP_TYPE'
```

根因：

* `rknn-toolkit2 2.3.2` 内部仍依赖 ONNX 较旧的 `mapping` 接口
* 而新版 ONNX 的接口结构已经调整

解决方式：

在 `convert.py` 中手动添加兼容层，补齐：

* `onnx.mapping`
* `TENSOR_TYPE_TO_NP_TYPE`
* `NP_TYPE_TO_TENSOR_TYPE`

这一点是本项目成功完成 `ONNX → RKNN` 的关键。

---

### 6.5 第五步：完成 ONNX → RKNN

在解决兼容问题后，成功运行 `convert.py`，完成：

```text
best.onnx → best.rknn
```

转换成功后，输出日志中还出现了中间优化模型：

* `check0_base_optimize.onnx`
* `check2_correct_ops.onnx`
* `check3_fuse_ops.onnx`

这些文件表明：

* ONNX 已被 RKNN 正常解析
* 图优化成功
* 构建成功
* 导出成功

最终得到：

```text
best.rknn
```

---

## 7. 板端部署准备

---

### 7.1 真正需要放到板端的内容

板端最小运行集合为：

```text
best.rknn
images/
infer_defect_folder.py
```

### 7.2 不需要放到板端的内容

以下文件不需要传到开发板：

* `best.onnx`
* `best.onnx.data`
* `dataset.txt`
* `dataset/`
* `check*.onnx`

原因：

* 这些都只在 PC 侧转换和量化时使用
* 板端推理只依赖 `.rknn`

---

## 8. 板端验证过程

---

### 8.1 板端已有环境检查

在板端检查后发现：

* `python3` 存在
* `python` 命令不存在
* `rknn-toolkit-lite2` 已安装但版本为 `1.6.0`
* `librknnrt.so` 存在
* `images/` 目录中有 50 张图片
* `best.rknn` 已在 `/home/cat`

所以后续板端脚本全部统一使用：

```text
python3
```

而不是 `python`。

---

### 8.2 旧脚本分析

板端原先已有两个旧脚本：

* `test_rknn.py`
* `yolo11_seg_run.py`

它们的问题主要有：

1. 使用了旧模型名
   例如：

   ```text
   best_yolo11_seg_rk3588.rknn
   ```

   而当前实际模型为：

   ```text
   best.rknn
   ```

2. 写死了单张图片
   而当前图片都放在：

   ```text
   /home/cat/images/
   ```

3. 不能满足批量处理需求

因此旧脚本没有继续沿用，而是重新编写了新脚本。

---

### 8.3 最小验证脚本

先编写了 `check_best_rknn_folder.py`，只做：

* 加载 `best.rknn`
* `init_runtime()`
* 从文件夹中读取一张图
* 推理
* 打印 output shape

板端实际输出如下：

```text
output[0]  = (1, 64, 80, 80)
output[1]  = (1, 1, 80, 80)
output[2]  = (1, 1, 80, 80)
output[3]  = (1, 32, 80, 80)

output[4]  = (1, 64, 40, 40)
output[5]  = (1, 1, 40, 40)
output[6]  = (1, 1, 40, 40)
output[7]  = (1, 32, 40, 40)

output[8]  = (1, 64, 20, 20)
output[9]  = (1, 1, 20, 20)
output[10] = (1, 1, 20, 20)
output[11] = (1, 32, 20, 20)

output[12] = (1, 32, 160, 160)
```

并且运行时板端打印出：

* runtime version: `1.6.0`
* toolkit version embedded in model: `2.3.2`

说明存在版本不一致警告，但当前模型可加载、可推理。

---

## 9. 输出结构分析

从实际输出形状可以推断：

### 每个尺度包含 4 组输出

#### 80×80 尺度

* `64` 通道：框回归 DFL
* `1` 通道：分数分支 1
* `1` 通道：分数分支 2
* `32` 通道：mask coeff

#### 40×40 尺度

同上

#### 20×20 尺度

同上

#### 最后一个输出

* `(1, 32, 160, 160)`
* 这是 segmentation 的 proto 特征图

---

## 10. 只做检测、不做分割的部署策略

由于当前目标只是 defect 检测，因此在板端后处理时采用以下策略：

### 10.1 保留的输出

* 框回归：`64`
* 分数分支：`1 + 1`

### 10.2 忽略的输出

* 每尺度的 `32` 通道 mask coeff
* 最后一个 `32×160×160` proto

也就是说：

* 模型结构没裁掉 seg head
* 但推理后处理阶段不去解 mask

这是本项目最有效、最省事的路线。

---

## 11. 板端后处理开发过程

---

### 11.1 第一版问题：满屏绿色框

最初在 `infer_defect_folder.py` 中，错误地将两个 `1` 通道分数分支做了组合，导致：

* 候选框数过多
* 大量错误框被画出
* 结果图看起来整张变绿

这说明：

* 不能盲目假设两个 `1` 通道都该直接乘起来参与分类

---

### 11.2 修正思路

后续采取的修正策略有：

1. 提高置信度阈值
2. 降低 NMS 阈值
3. NMS 前限制候选框数量
4. 暂时只使用 `score1`
5. 忽略 `score2`

最终采用的参数是：

```text
CONF_THRESH = 0.60
NMS_THRESH  = 0.30
```

并在 NMS 前仅保留 top 200 候选框。

---

### 11.3 预处理方式

使用 letterbox：

* 按比例缩放
* 补边到 640×640
* BGR → RGB
* 输入格式为 NHWC

即最终输入形状：

```text
(1, 640, 640, 3)
```

---

### 11.4 检测解码方式

采用：

* `REG_MAX = 16`
* `STRIDES = [8, 16, 32]`

步骤包括：

1. 三个尺度分别解码
2. 对 `64` 通道 DFL 做 softmax 解码
3. 根据 stride 恢复边框
4. 去除 letterbox padding
5. 映射回原图坐标
6. 合并三尺度结果
7. 做 NMS

---

## 12. 最终脚本功能

最终版 `infer_defect_folder.py` 实现了以下功能：

* 自动遍历 `/home/cat/images`
* 支持 `.jpg/.jpeg/.png/.bmp/.webp`
* 加载 `/home/cat/best.rknn`
* 对每张图做缺陷检测
* 忽略 segmentation 输出
* 仅绘制 defect 检测框
* 保存结果图到 `/home/cat/results`

最终结果图中，已经可以看到：

* 缺陷附近有合理的小框
* 框不再满屏泛滥
* 每张图有 defect 分数标注

这说明当前部署链路已经能够支撑：

```text
离线图片批量缺陷检测
```

---

## 13. 本项目已验证成功的能力边界

当前已经验证成功的是：

* RK3588 上加载 `.rknn`
* 文件夹批量图片推理
* defect 检测框输出
* 可视化结果图保存

当前尚未做的是：

* segmentation mask 后处理
* 板端实时摄像头接入
* 检测结果结构化保存为 CSV / TXT
* runtime / toolkit 版本完全对齐
* 多类别场景扩展

---

## 14. 本项目踩坑总结

### 14.1 版本比代码更重要

本项目最大的经验之一是：

很多问题不是模型本身错了，而是环境版本不兼容。

尤其是：

* Python 版本
* ONNX 版本
* RKNN Toolkit 版本
* RKNN Lite 版本
* 板端 Runtime 版本

---

### 14.2 不要一开始就假设后处理逻辑

对于自训练模型，尤其是：

* seg 模型
* 自定义导出链路
* RKNN 优化后的输出

不能直接套网上现成 YOLO 后处理代码。

必须先做：

1. 打印输出 shape
2. 分析每个输出含义
3. 再决定如何解码

---

### 14.3 分割模型可以先只做检测

如果当前业务只需要检测框，那么最实用的路线是：

* 先保留 seg 模型结构
* 板端只取 detect 部分
* 暂时不解 mask

这样可以显著降低部署复杂度。

---

### 14.4 图片文件名不固定时必须遍历文件夹

实际业务图片通常文件名复杂且不规则，不能写死：

```text
test.jpg
```

应统一采用：

```text
遍历 images 目录
```

---

### 14.5 板端必须以 `python3` 为准

本项目板端系统中：

* `python3` 存在
* `python` 不存在

因此板端所有脚本都应以：

```text
python3 script.py
```

执行。

---

## 15. 当前遗留风险

虽然当前流程已经跑通，但还存在一个明确风险：

```text
RKNN model toolkit version 2.3.2
runtime version 1.6.0
```

这意味着：

* 模型导出版本较新
* 板端 runtime 较旧

当前能跑，但不代表长期一定稳定。

后续如果进入更正式部署阶段，建议升级板端：

* `rknn-toolkit-lite2`
* `librknnrt.so`

尽量与模型导出版本靠近。

---

## 16. 这套流程的推荐复用模板

以后做同类项目时，建议统一按照以下顺序：

### 16.1 PC 侧

1. 导出 ONNX
2. 校验 ONNX
3. 准备量化集
4. 固定版本环境转换 `.rknn`

推荐固定环境：

```text
Ubuntu 24.04
Python 3.11.9
rknn-toolkit2 2.3.2
onnx 1.18.0
onnxruntime 1.18.0
numpy 1.26.4
```

### 16.2 板端

1. 先验证 `load_rknn()`
2. 再验证 `init_runtime()`
3. 打印 output shape
4. 按实际输出写后处理
5. 先跑图片批量推理
6. 稳定后再接摄像头

### 16.3 业务策略

1. 先跑通再优化
2. 先离线图片再实时视频
3. 先检测再考虑分割
4. 先做最小闭环再扩展功能

---

## 17. 本项目结论

本项目最终成功实现了：

* YOLO11n-seg 模型转换为 RKNN
* RK3588 / 鲁班猫板端加载推理
* 忽略 segmentation，仅做 defect 检测
* 批量处理图片文件夹
* 输出检测框结果图

这是一个已经具备复用价值的完整闭环。

对于后续同类型项目，这份归档文档可以直接作为：

* 项目起步模板
* 环境版本参考
* 踩坑记录
* 部署流程标准

---

如果你愿意，我下一条可以继续帮你把这份“详细项目归档版”再整理成 **README.md 风格版本**，方便你直接放进仓库。
