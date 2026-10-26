# multiref-eval

基于 MultiRef-Compass 原评分规则的本地裁判适配工程。第一阶段保留原四个维度、14 项指标、原分值及汇总；不新增百分制或统一总分。

## 代码结构

```text
finaleval/                       原官方计算代码，保持原目录与公式
  basic_quality/                VQ / AQ / 解剖合理性
  entity_fidelity/              EF / DP / DB 与贴图辅助项
  audio_video_consistency/      口型、音画事件、声音来源、音色
  instruction_following/        视觉、声音、时序、台词遵循
  common/                      输入结构与原输出工具
multiref_eval/                  本工程入口、输入验证、环境检查
configs/                       指标列表与本地配置模板
scripts/                       原脚本及本工程 run_eval.sh
docs/                          上游文档与适配说明
tests/                         不需要模型的回归测试
upstream.lock.json             来源提交及原文件 SHA256
```

## 当前状态

已导入原评分实现并建立工程入口，不代表完成原论文复现。`finaleval/` 保留上游计算；仅修复一处 Python 3.10 报告文本语法，记录在 `upstream.patches.json`，其余已知接线问题尚待处理。

本工程 `evaluate` 只开放 VQ、AQ、TS 的原后端调用，仍需配置依赖和权重并实际验证。VQ/AQ 共用原基础质量流程，运行其中一项也可能需要另一项的环境。

其他 11 项在本工程入口明确报未就绪：Qwen/Qwen-Omni 媒体输入、提示词及结构化输出适配尚未实现；EF/SLS 导出接线也需要核对。不自动回退 Gemini。直接运行上游 `finaleval.cli` 或原 shell 脚本会绕过该限制，可能调用 Gemini，暂勿作为本工程评测入口。

## 无模型检查

在仓库根目录执行，Python 3.10+，仅标准库即可：

```bash
python3 -m multiref_eval list-metrics
python3 -m multiref_eval doctor --config configs/local.example.json
python3 -m multiref_eval audit-dataset --root /mnt/AINAS0/user/xingshen/MultiRef-Compass
python3 -m unittest discover -s tests -v
```

数据集为 350 道题。公开版有 32 道题提及参考音频但未附素材；审计只报告，不改题、不补造音频。数据集不含待测模型生成结果，必须先生成视频才能评分。

## 接入评测

复制 `configs/local.example.json` 为 `configs/local.json`，配置实际缓存路径。裁判字段目前仅记录计划模型，不能据此启用 Qwen。

输入沿用原 CSV：至少 `sample_id,model_name,prompt,video_path`；参考图使用 `reference_image_paths` JSON 数组，其他列参见 `docs/upstream_README.md`。TS 还要求真实的 `reference_audio_path`。输入路径相对 CSV 所在目录解析。

```bash
python3 -m multiref_eval validate --input /absolute/path/generated.csv --metrics visual_quality
# 配置好相应后端环境和权重后才可运行：
python3 -m multiref_eval evaluate --input /absolute/path/generated.csv --metrics visual_quality --config configs/local.json --output /absolute/path/new-run
```

可用 `python3 -m pip install -e .` 安装命令行入口；评分依赖需另按 `docs/model_weights.md` 逐项准备，`pip install -e '.[scoring]'` 仅覆盖基础依赖，不是完整后端环境。尚未完成评分环境安装或 GPU 评测。权重已下载至 aigc-18 的 NAS，文件清单见 configs/model_assets.json。

## 来源与许可

官方仓库：https://github.com/zxhhh0201/MultiRef-Compass

固定提交：`03da121e88e241da38b41324291dfa7b8773d132`。保留原 MIT LICENSE，来源及文件哈希见 `upstream.lock.json`。模型权重各自许可独立，不由本仓库 MIT 许可覆盖。

## 北京 3090 机器部署

参见 [部署交接说明](docs/DEPLOY_BEIJING_3090.md)，包括代码就绪边界、数据和权重位置、现有裁判服务，以及待完成的适配。
