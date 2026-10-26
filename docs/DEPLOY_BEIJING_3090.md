# beijing-4-3090-2 部署交接

## 当前边界

本仓库是原评分代码与适配工程基础版，不是已完成部署的全指标服务。部署方式为 GPU 批处理；Qwen / Omni 复用 HTTP 服务。原评分公式、量表和封顶保持。

- 已导入四维度、14 指标原代码，新增输入验证、数据审计、环境诊断与错误检查。
- 新入口仅开放 VQ、AQ、TS 原后端；仍需要环境、权重和端到端验证。
- 其余指标在新入口明确阻止运行：Qwen 媒体适配、EF/SLS 输出字段尚未接通。不要改为直接调用原 Gemini 入口来绕过。
- aigc-18 系统 Python 3.10 缺 ffmpeg、torch 等依赖。之前尝试创建的 .venv 未完成，不是可复制的运行环境。

## 数据与权重

源服务器 aigc-18：

```text
/mnt/AINAS0/user/xingshen/MultiRef-Compass
/mnt/AINAS0/user/xingshen/multiref-eval-models
```

先确认北京机器是否挂载同一 NAS；不能仅凭路径同名假定文件可读。未挂载则将素材与权重另行传输，勿提交 Git。

已完成下载并进行大小 / 哈希检查：DOVER、DOVER++、Audiobox、SyncNet 及辅助文件、ECAPA、SigLIP、DINOv2、OpenS2V 的 YOLO-World adapter / CurricularFace / buffalo_l、YOLOv8 动漫脸，以及 ElasticFace-Arc。

ElasticFace 明确选用官方 Arc IResNet-100（MS1MV2）；这是用户授权的适配选择，不能宣称确认了论文作者的原始版本。详情见 configs/model_assets.json。可选 MMDet3 YOLOv3 未补齐；原 shell 默认使用已下载的 YOLOv8 动漫脸。所有模型尚未完成加载与推理验证。

## 初始化

```bash
git clone https://github.com/XINGSHEN2/multiref-eval.git
cd multiref-eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp configs/local.example.json configs/local.json
python -m multiref_eval doctor --config configs/local.json
python -m unittest discover -s tests -v
```

如缺 venv/ensurepip，先准备可用的 Python 环境。编辑 local.json 的真实数据、权重及缓存路径。基础安装仅提供 CLI；GPU PyTorch、ffmpeg 和各后端依赖要按机器驱动及原实现分别安装验证，不能将此命令理解为评分环境已齐备。权重路径对应关系见 docs/model_weights.md 与 configs/model_assets.json。

## 设备与验证顺序

先检查 nvidia-smi 和已有 GPU 任务；不得为部署停止现有服务。四张 3090 可按 VQ / EF / SyncNet / 音频指标分组调度，但应先单指标跑通，再并行。每个模型加载一次后连续处理样本。

1. 验证媒体读取和专用模型加载，使用少量已有视频，不重新制作测试集。
2. 依次跑通 VQ、AQ、TS，再接 EF / SLS。VQ/AQ 原后端共用流程，可能同时依赖两套环境。
3. 接入 Qwen 的实际图片 / 帧输入和 Omni 的音轨 / 视频输入，保持原问题与计分。
4. 保存逐样本分数、覆盖率、错误及实际版本，再开始全量评测。

## 现有裁判服务（2026-10-26 从 aigc-18 核验）

- 视觉主候选 http://10.0.96.114:9012，Qwen3.8-27B。
- 视觉备选 http://10.6.157.43:9012，Qwen3.8-27B。
- 音频/音画 http://10.6.157.42:9013，Qwen3-Omni-30B-A3B-Instruct。

上述 /health、/v1/models、/parameters 当时可访问，未做推理验证。北京机器需重新检查网络。视觉接口最多 8 张图或单视频，不接受图片和视频混合，也不分析音轨；Omni 接口支持单音频或单视频，时长上限 30 秒。多参考输入的编排仍待实现，不得悄悄丢弃参考。
