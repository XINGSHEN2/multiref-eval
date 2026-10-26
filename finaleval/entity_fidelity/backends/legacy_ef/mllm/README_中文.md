# MLLM 评测 xlsx 入口

这个目录保留 anatomical consistency、detail preservation、binding、paste artifact 等 MLLM 评测需要的文件。

## 输入 xlsx 格式

必须包含这些列：

- `model`：生成视频的模型名，例如 `wan2.1`、`skyreels-new`
- `video_path`：要评测的视频绝对路径
- `board`：`board1`、`board2`、`board3`
- `seq`：样本编号，例如 `1` 或 `001`

也可以用 `index` / `sample_id` / `sample` 写成 `MultiOmni-board1-001`，脚本会自动解析 board 和 seq。

## 运行命令

在 `finalef` 目录执行：

```bash
XLSX=/path/to/input.xlsx BOARD=board1 bash run_mllm_detail_binding_from_xlsx.sh
```

单独跑某一项：

```bash
XLSX=/path/to/input.xlsx BOARD=board1 bash run_mllm_anatomical_from_xlsx.sh
METRIC=detail BOARD=board1 bash run_mllm_detail_binding_from_xlsx.sh
METRIC=binding BOARD=board1 bash run_mllm_detail_binding_from_xlsx.sh
BOARD=all bash run_mllm_paste_artifact_from_xlsx.sh
```

贴图感评测默认使用顶层的：

```text
final_videos_9models_board2_board3_mllm_detail_binding_input.xlsx
```

它会判断主体/物体是否有贴图、抠图、纸片感、边缘 halo、滑动层、光影/透视/接触阴影不融合等问题。这个分数用于后续 EF 惩罚，不评价 prompt following、reference 相似度、动作正确性或美观程度。

## 常用参数

- `BOARD=all|board1|board2|board3`：选择 board，默认 `all`
- `LIMIT_ROWS=10`：只从 xlsx 取前 10 行，方便小样本测试
- `ALLOW_MISSING_VIDEOS=1`：允许视频路径不存在，只生成 manifest 或继续走脚本
- `OIS_MAX_WORKERS=4`：anatomical 并发
- `DP_MAX_WORKERS=12`：detail preservation 并发
- `BINDING_MAX_WORKERS=12`：binding 并发
- `PASTE_MAX_WORKERS=12`：paste artifact 并发
- `OIS_EVAL_MODEL` / `DP_EVAL_MODEL` / `BINDING_EVAL_MODEL`：指定 Gemini judge 模型
- `PASTE_EVAL_MODEL`：指定 paste artifact 的 Gemini judge 模型

## 输出

- manifest：`mllm/inputs/mllm_<board>_<timestamp>.json`
- anatomical：`mllm/outputs/anatomical_consistency/`
- detail preservation：`mllm/outputs/detail_preservation/`
- binding：`mllm/outputs/binding/`
- paste artifact：`mllm/outputs/paste_artifact/`

每个评测脚本会自动写 summary CSV / detail CSV / ranking 或 per-sample 文件，最新输出目录记录在对应 `latest_run.txt`。


Detail/Binding 统一入口：`METRIC=both BOARD=board1 bash run_mllm_detail_binding_from_xlsx.sh`；单独跑用 `METRIC=detail` 或 `METRIC=binding`。

## Paste Artifact 输出含义

`paste_artifact_per_sample.csv` 里最重要的字段：

- `Paste Artifact Quality Score`：1-5 分，越高表示贴图感越少、融合越自然
- `paste_penalty`：用于乘到 EF 上的惩罚系数
- `severity`：`none|minor|moderate|severe|critical`
- `reason` / `comments`：模型给出的具体贴图证据

推荐后处理公式：

```text
FinalEF = mean(Human_EF_with_LTC, Object_EF_with_LTC, Scene_EF_with_LTC) * paste_penalty
```

生成最终 EF + 贴图惩罚汇总表的脚本在：

```text
finalef/summary_py/generate_ef_paste_summary.py
```

具体命令见顶层 `README_中文.md` 的“EF + 贴图惩罚汇总”章节。
