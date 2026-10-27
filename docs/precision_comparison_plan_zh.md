# 同模型不同精度输出对比方案

状态：设计方案，尚未实现 compare-pairs 命令，尚未产生质量评测结果。更新：2026-10-27。

## 目的与边界

在同一生成模型内比较 BF16、FP16、FP8 等推理配置，分别回答“输出变化多少”和“生成质量是否变差”。BF16 是对比基线，不是真实画面的标准答案。PSNR、SSIM、LPIPS 不转换成“画质损失百分比”。保持 MultiRef-Compass 原 14 项指标和汇总口径不变，新模块作为独立回归分析工具。

当前仓库 visual_quality 使用 DOVER++ technical；multiref_eval CLI 仅放行 visual_quality、audio_quality、voice_timbre_similarity 的后端调用，仍需环境与实际推理验证。其他裁判/指标按原就绪检查处理，不绕过入口调用 Gemini。参见 IMPLEMENTATION_STATUS.md。

## 对比设计与输入

建议新增 pair manifest（JSONL），每行一对输出，字段包括：

- pair_id、sample_id、seed、模型名称及源 checkpoint revision/hash。
- baseline_precision、candidate_precision，以及两份视频的绝对路径或相对 manifest 路径、SHA256。
- 官方 prompt/IR hash、参考素材及顺序的 hash、生成参数（尺寸、帧率、步数、scheduler、guidance 等）。
- 代码/镜像版本、实际量化目标、权重/激活 dtype、量化 scale 策略；不能只记录 fp8 标签。
- 初始化噪声 hash（若有）、GPU 型号/组、并发、预热与编码设置；缺失字段明确为 unknown。

同案例、同 seed 配对；同源 checkpoint 的量化衍生权重允许 hash 不同，但须记录来源。IR、素材或生成参数不一致时拒绝进入受控比较；运行环境差异可保留为 observational，并展示差异。相同 seed 不保证跨设备/内核得到完全相同的噪声或结果。

一个候选对应一个指定基线。重复测量必须记录 repeat，不覆盖或自动挑选最接近的输出。缺少一侧视频输出 missing_pair；解码失败输出 decode_error，不补零、不计为质量下降。

## 两组互补指标

| 组别 | 指标 | 解释 |
|---|---|---|
| 配对差异，第一阶段 | MSE、PSNR、SSIM | 同时间位置画面的像素/结构变化；CPU 可计算，无模型权重 |
| 配对差异，可选 | LPIPS | 预训练视觉特征的感知差异；越小越相似，不等于越高质量 |
| 独立质量 | DOVER++ technical | 每段视频独立评分，报告 candidate - baseline；越高越好 |
| 音频，可选 | 原 audio_quality、voice_timbre_similarity | 使用原适用性规则；音色比较需要真实参考音频，不把 BF16 音轨冒充参考素材 |
| 人工复核 | 随机 A/B 盲评 | 检查提示词遵循、参考一致性、画面缺陷、时序连续性、音频与同步 |

LPIPS、DOVER 等评估器须固定相同版本、权重、输入预处理与评估精度，不能 BF16 输出用一套裁判、FP8 输出用另一套。评分器随机抽样应固定并记录 seed，必要时重复评分估计评分波动。

## 解码、对齐与计算约定

1. 使用 ffprobe 保存两侧 codec、尺寸、pix_fmt、色彩矩阵/range/transfer、帧率、time_base、帧数、PTS 和时长。
2. 默认严格比较：相同尺寸，按相对展示时间戳配对；时间容差明确记录（建议不超过半个帧间隔），不允许重复配对。VFR 或不匹配输入报告不兼容，不静默截短、重采样、插帧或缩放。
3. 默认将两侧用同一解码流程转换为 RGB24，再转 float [0,1]；保存 FFmpeg 版本与转换参数。色彩元数据缺失时记录假设，HDR 不直接按 SDR 处理。可另做 Y 通道报告，禁止混为同一指标。
4. PSNR：逐帧 MSE 为所有 RGB 像素平方误差的平均；PSNR = 10 log10(1/MSE)。MSE=0 时记录 exact_match 与 +infinity；JSON 用 null 加标记表示，不写非标准 Infinity。整段主 PSNR 由全段累计 SSE/元素数得到，不直接把 dB 的帧均值冒充整段 PSNR。
5. SSIM：建议固定 scikit-image 实现，RGB channel_axis=-1、data_range=1、Gaussian window 11、sigma=1.5、use_sample_covariance=False；锁定版本，尺寸不足时显式报错。每帧输出分数，汇总均值与低分分位数。
6. LPIPS：建议第一版固定 AlexNet backbone、LPIPS v0.1、eval/no_grad，RGB 转 [-1,1]，固定空间平均规则。默认按原分辨率、小 batch 推理；可选缩放/抽帧须单独标注，不能与全帧原尺寸结果混算。
7. 像素指标默认全帧流式计算，避免一次加载整段视频。LPIPS 可先按固定时间点抽帧，并明确抽样覆盖率。
8. 已有 MP4 的差异包含编码影响，不能全部归因于模型量化。以后若重新生成，优先另存未压缩帧或一致的无损中间输出；本次不因评测方案而重跑生成。
9. 不做光流补偿、自动裁剪或时间搜索来改善主指标。必要的对齐实验作为单独诊断，保留原始严格比较结果。

## 输出与结论

建议新增 multiref_eval/precision_comparison/，复用输入路径解析与来源记录；新增独立 compare-pairs 子命令，不将指标加入原 14 项总分。以下文件名为拟定接口，不代表目前存在可运行实现。

- run.json：版本、参数、输入/权重 SHA、解码约定、执行设备、时间、完整性状态。
- per_frame.csv：pair_id、两侧帧号/PTS、MSE、PSNR、SSIM、可选 LPIPS、exact_match。
- per_pair.csv：覆盖率、整段 PSNR、SSIM 均值/P05、LPIPS 均值/P95、状态及原因。
- quality_delta.csv：原指标的两侧原始分和 candidate - baseline、适用性与错误状态；保持分数原单位，不统一成百分制。
- report.html：A/B 视频、逐帧曲线、最差帧及其相邻帧、差异热图。注明热图放大倍率、量程及选帧依据，不把放大误差图当实际可见失真。

首先逐例报告，不仅给全局平均。跨案例汇总默认每对等权，另报缺失覆盖率，避免长视频支配结果。多 seed 后可报告配对差值的置信区间，以案例/seed 为统计单元，不能把同视频的数百帧当独立样本。条件允许时加入 BF16 重复生成以估计自身非确定性；现有结果没有该对照时明确限制。

不设通用的 PSNR/SSIM 及格阈值。先用人工盲评建立可接受样例与严重缺陷清单，再预先约定业务阈值与置信区间规则。DOVER 分数差也不是感知质量下降百分比。

## 权重与资源需求

| 方法 | 额外模型权重 | 设备建议 |
|---|---|---|
| ffprobe/解码、MSE、PSNR、SSIM、差异图 | 无；仅需 FFmpeg 与数值/图像库 | CPU |
| LPIPS | 需要选定 backbone 的预训练权重，以及 LPIPS 校准权重（部分随包分发）；首次运行可能触发下载 | CPU 可运行，单卡更快 |
| DOVER++ | 需要 DOVER++ checkpoint 与后端依赖；当前仓库 setup 也准备 DOVER.pth | 推荐单卡，具体显存由预处理/batch 实测 |
| audio_quality | 当前 BQ 路径涉及 AudioBox checkpoint，且与 DOVER 流程耦合 | 依原后端环境；不是只装一个包即可 |
| voice_timbre_similarity | SpeechBrain ECAPA-TDNN 及其配置/辅助文件 | CPU 或 GPU |
| 人工 A/B | 无 | 普通视频播放设备 |

对现有视频做以上评估不需要 H3 生成模型权重，也不需要重新生成。只有重跑生成或比较未保存的中间 latent/预测时才需要加载 H3。

已有权重清单位于 configs/model_assets.json，声明 aigc-18 的缓存根目录为 /mnt/AINAS0/user/xingshen/multiref-eval-models，包含 dover/DOVER.pth、dover/DOVER_plus_plus.pth、audiobox-aesthetics/checkpoint.pt、spkrec-ecapa-voxceleb 等。清单 inference_verified=false，且路径未自动接入所有 legacy 后端。先核对文件、hash 和加载路径，再决定是否下载；本方案不代表推理已验证。清单未列出 LPIPS，实施时单独检查本机缓存，不能假定已经具备。

权重置于仓库外，记录官方来源、版本、许可与 SHA256；优先使用已有已验证缓存。提供离线/禁止自动下载选项，缺失时明确报 missing_weights，不运行时悄悄联网。单卡评分不等于 H3 生成所需的八卡 worker。

## 分阶段落地与验收

阶段一：实现 manifest 校验、媒体预检、严格配对、CPU PSNR/SSIM、逐帧 CSV 和 HTML。无额外模型权重。

阶段二：可选 LPIPS，接入已有 DOVER++ 两侧分数与质量差值。先验证后端实际加载/推理，复用缓存，不绕过当前 CLI 就绪限制。

阶段三：多案例多 seed、人工盲评、必要时基线重复生成；补适用的音频和时序指标，不用数值相似度替代语义评估。

实现验收应包含：相同解码帧的 MSE=0/PSNR=+infinity/SSIM≈1；已知像素扰动的解析 PSNR；帧错位、分辨率不匹配、截短/坏视频、重复配对、缺少权重的显式失败；缺失项不污染均值；逐帧最差位置与可视化一致。LPIPS 增补同输入近零及固定版本重复性检查。

本次 H3 suite-20261023 有 7 对完整 FP8/BF16 视频。1.1 case5 只有 FP8；3.4 case3 两种精度均因原机器到期中断，均排除配对分析。1.1 FP8 有三次历史正式结果，本地展示为第 1 次：像素比较须固定该实际视频，不能将三次耗时中位数理解成存在一段“中位数视频”。历史/并发条件差异作为限制保留。用户已决定不补跑，第一阶段只读取现有输出。
