# PureAC VSLS Pipeline

本文档说明当前 PureAC VSLS 使用的最终逻辑。核心思路是：先用 MLLM 筛出适合自动口型同步评测的视频，再用 VSLS/SyncNet 对这些视频做自动化打分；多人脸场景默认按最低 LSE-D 选择说话人脸轨迹。

## 1. 整体流程

PureAC VSLS 分为 MLLM 前置筛选和自动化 VSLS 两部分。

1. 从输入 CSV 读取视频路径、模型名、样本 ID、board、prompt、`has_speech` 等信息。
2. 如果启用 `stable_speaking_frontal_face`，先只在有 speech 的样本上调用 MLLM。
3. MLLM 判断视频是否满足“说话时稳定正脸、半身或更近、嘴部可见、说话人可识别”等条件。
4. 只保留 MLLM 判定可用于 VSLS 的样本，生成 `stable_speaking_frontal_input.csv`。
5. 自动化 VSLS 对筛选后的视频做语音检测、人脸检测、多人脸轨迹选择和 SyncNet 打分。
6. VSLS 后处理再做一次 crop 级别 stable-frontal 检查。
7. 最终只对通过 MLLM 筛选、SyncNet 有效、offset 合理、crop 稳定正脸的窗口汇总分数。

## 2. MLLM 在 VSLS 中的作用

MLLM 不直接给 lip-sync 分数。它的作用是筛掉不适合用自动化 SyncNet 评测的视频，避免远景、侧脸、背脸、口部遮挡、off-screen speech、多人说话人不明确等样本污染自动指标。

当前 VSLS 前置筛选使用的 MLLM 指标是：

```text
Stable Speaking Frontal Face
```

该指标只看生成视频和音频，重点观察有 speech 的时间段。通过条件包括：

- 有可听 speech 或 voice；
- 说话人出现在画面中；
- 说话时为半身、上半身、bust-style 或更近景别；
- 说话期间大部分时间是正脸或近正脸；
- 嘴部或下半脸足够可见，能够支持 lip-sync 评估；
- 人脸不能太小、严重模糊、严重遮挡、频繁出画或主要侧脸；
- 镜头和人脸轨迹足够稳定，不能频繁切走、剧烈运动或尺度大幅变化；
- 多人场景中说话人必须能可靠识别。

导出 VSLS 输入 CSV 时，只保留：

```text
applicable = true
stable_speaking_frontal_face = true
score >= 4
```

输出文件通常为：

```text
outputs/<run_name>_<timestamp>/stable_speaking_frontal_input.csv
outputs/<run_name>_<timestamp>/stable_speaking_frontal_rejected.csv
```

其中 `stable_speaking_frontal_input.csv` 是后续 VSLS 自动化指标的输入。

## 3. 其他 PureAC MLLM 指标

PureAC 里还有两个 MLLM 音频一致性指标，它们和 VSLS 是并列关系，不直接参与 SyncNet 分数计算。

### Visual Event-Sound Alignment

该指标判断非语音音效和画面事件是否匹配，例如撞击、脚步、开门、水声、物体运动声等。它不评估 speech、背景音乐或 lip-sync。

当前规则中，off-screen 环境声如果与场景合理一致，不扣分；匹配的声音如果只是 slightly quiet 或 somewhat masked，也不扣分。

### Source Correctness

该指标判断 audible speech/voice 是否来自正确的可见说话人。它不评估台词是否逐字正确、不评估语言是否正确、不评估 lip-sync 时间对齐，也不评估人物 identity 相似度。

如果声音是旁白、off-screen voiceover、说话人不可见、嘴部不可见、人物太小或多人说话人不明确，则通常标为 not applicable。

## 4. 自动化 VSLS 预处理

VSLS 自动化部分使用 LatentSync/SyncNet。每个视频先被标准化：

```text
video: 25 fps H264
audio: 16 kHz mono wav
```

如果音频明显长于视频，会按视频时长裁剪；如果视频长于音频，则使用两者共同的有效时长。

之后使用 VAD 检测 speech 片段。默认 VAD backend 是：

```text
silero
```

主要参数为：

```text
min speech duration: 0.30 s
merge gap: 0.20 s
threshold: 0.5
```

检测到的 speech 会被切成约 2 秒的非重叠窗口。后续人脸检测和 SyncNet 打分都在这些 speech windows 内进行。

## 5. 人脸检测与轨迹构建

每个 speech window 内使用 LatentSync 的 S3FD 人脸检测器检测人脸，并基于 IoU 构建人脸轨迹。

当前主要条件为：

```text
face detector: S3FD
min face confidence: 0.90
track IoU threshold: 0.30
max track gap: 5 frames
min track frames: 20
speech-window coverage >= 0.80
mean face size >= 64 px
```

每条候选人脸轨迹会被裁成 face crop。裁剪方式是以人脸框为中心做 expanded square crop：

```text
crop scale: 1.30
crop size: 224 x 224
```

然后把该 face crop video 和对应 speech-window audio 一起送入 SyncNet。

## 6. SyncNet 输出

SyncNet 对每个候选人脸轨迹输出三个核心指标：

```text
LSE-C: confidence，越高越好
LSE-D: distance，越低越好
offset_frames: 音画偏移帧数，绝对值越小越好
```

其中 LSE-C 反映同步置信度，LSE-D 反映音画嵌入距离，offset_frames 反映估计出的音画时间偏移。

## 7. 多人人脸选择逻辑

当前固定使用后来的逻辑：每个 speech window 内，先过滤掉 offset 过大的候选轨迹，再按最低 LSE-D 选择说话人脸。

具体规则为：

```text
候选过滤: abs(offset_frames) < 10
主排序: LSE-D 最低
第一 tie-break: abs(offset_frames) 更小
第二 tie-break: LSE-C 更高
```

也就是说，多人场景里不再默认选综合分最高的 track，而是优先选择 SyncNet distance 最低的 track。这样更符合后续固定逻辑：用 LSE-D 最小的脸作为该 speech window 的同步对象。

## 8. VSLS 分数计算

选定人脸轨迹后，将 LSE-C 和 LSE-D 归一化，再计算 0-1 分和 1-5 分。

公式为：

```text
C_norm = clamp((LSE-C - 1.5) / (7.0 - 1.5), 0, 1)
D_norm = clamp((13.0 - LSE-D) / (13.0 - 7.0), 0, 1)

Score_0_1 = 0.60 * C_norm + 0.40 * D_norm
Score_1_5 = 1 + 4 * Score_0_1
```

因此最终 VSLS 分数同时考虑 LSE-C 和 LSE-D，其中 LSE-C 权重为 60%，LSE-D 权重为 40%。

这里的两个归一化分数含义如下：

- `C_norm` 是由 LSE-C 转换得到的同步置信度分数。LSE-C 本身越高越好，所以 `C_norm` 也越高越好；当 LSE-C 小于等于 1.5 时记为 0，当 LSE-C 大于等于 7.0 时记为 1，中间线性缩放。
- `D_norm` 是由 LSE-D 转换得到的同步距离分数。LSE-D 本身越低越好，所以这里做了反向归一化，把“距离越小越好”转换成“分数越高越好”；当 LSE-D 大于等于 13.0 时记为 0，当 LSE-D 小于等于 7.0 时记为 1，中间线性缩放。
- `clamp(..., 0, 1)` 表示把结果限制在 0 到 1 之间，低于 0 的按 0 计，高于 1 的按 1 计。

对每个有效 speech window，最终窗口级 VSLS 1-5 分数就是：

```text
VSLS_window_1_5 = 1 + 4 * (0.60 * C_norm + 0.40 * D_norm)
```

其中：

```text
C_norm = 0 表示 LSE-C <= 1.5
C_norm = 1 表示 LSE-C >= 7.0
D_norm = 0 表示 LSE-D >= 13.0
D_norm = 1 表示 LSE-D <= 7.0
```

所以高分需要同时满足较高的 LSE-C 和较低的 LSE-D。`Score_1_5` 的理论范围是 1 到 5，越高表示音画同步越好。

## 9. Stable-Frontal 自动复核

MLLM 通过后，自动化后处理还会对选中的 face crop 再做一次 stable-frontal 检查。该检查使用 OpenCV Haar frontal-face cascade，在每个 crop video 中抽取 8 帧。

通过条件为：

```text
frontal-face hit rate >= 75%
mean face width >= 28% of crop
center std <= 0.12
size CV <= 0.35
```

常见拒绝原因包括：

```text
missing_crop
low_frontal_detection_rate
face_too_small
face_center_unstable
face_size_unstable
```

最终 VSLS 排名只保留 stable-frontal 通过的 crop windows。

## 10. 最终过滤与聚合

最终进入模型排名的窗口必须同时满足：

```text
MLLM stable_speaking_frontal_face 通过
VSLS result 中 video_score_0_1 非空
speech segment status = scored
候选 track 满足 abs(offset_frames) < 10
按最低 LSE-D 选出有效 track
selected crop 通过 stable-frontal 自动复核
```

聚合方式为 video-macro：

1. 对同一个 `(model, sample_id)` 下的多个 speech windows 求平均，得到视频级分数。
2. 对同一个 model 下的多个视频级分数求平均，得到模型分数。
3. 同时输出 overall、board1、board2、board3 四个 scope 的排名。

也就是说，最终表里的模型 VSLS 1-5 分数按下面两步计算：

```text
VSLS_video_1_5(model, sample_id)
  = mean(VSLS_window_1_5 for all kept windows of this video)

VSLS_model_1_5(model)
  = mean(VSLS_video_1_5 for all kept videos of this model)
```

这里的 `kept windows` 必须已经通过最低 LSE-D track 选择、`abs(offset_frames) < 10` 过滤和 stable-frontal crop 复核。这样做的结果是每个视频先贡献一个视频级分数，模型最终分数不是直接对所有窗口做全局平均，而是对视频级分数做 macro average，避免 speech windows 多的视频在模型均分里占过大权重。

输出表中主要字段含义为：

```text
Score 1-5: 最终 VSLS 分数，越高越好
LSE-C: 选中窗口的平均 confidence，越高越好
LSE-D: 选中窗口的平均 distance，越低越好
Abs Offset: 选中窗口的平均绝对 offset，越低越好
Videos: 进入统计的视频数量
Windows: 进入统计的 speech windows 数量
```

## 11. 与 MLLM 指标的关系

PureAC 中的 MLLM 和自动化 VSLS 是互补关系。

MLLM 负责判断语义和可评测性，例如声音来源是否正确、事件音效是否符合画面、说话时人脸是否适合做口型同步评测。自动化 VSLS 则只在筛选后的 speech windows 上测量音画同步质量，主要依赖 SyncNet 的 LSE-C、LSE-D 和 offset。

因此，VSLS 最终分数不等价于完整的 Audio Consistency 分数。它更准确地表示“在可见、稳定、正脸说话条件下，模型生成的视频口型和音频 speech 是否同步”。

## 12. 推荐报告描述

可以在论文或报告中这样描述该流程：

> For speech-lip synchronization, we first use an MLLM-based pre-filter to retain only videos in which audible speech is produced by a visible speaker with stable frontal or near-frontal face visibility and sufficient mouth visibility. On the retained videos, we apply a SyncNet-based VSLS pipeline to speech windows detected by VAD. In multi-person scenes, candidate face tracks are first filtered by temporal offset, and the track with the lowest LSE-D is selected as the speaking face, with lower absolute offset and higher LSE-C used as tie-breakers. The selected track is further verified by a crop-level stable-frontal check. For each valid speech window, LSE-C is normalized as a confidence score and LSE-D is reverse-normalized as a distance score, and the final window score is computed as `VSLS_1_5 = 1 + 4 * (0.60 * C_norm + 0.40 * D_norm)`. 
