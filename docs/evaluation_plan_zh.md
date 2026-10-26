# 多参考视频生成评测方案：MultiRef-Compass 原评分方法与开源裁判适配版 v1

> 更新日期：2026-10-23。  
> 状态：方案已按本次讨论调整；尚未部署模型或运行评测。  
> 当前用途：非商业测试；使用可本地部署的公开模型替换闭源裁判。  
> 最新决定：第一版沿用原版各指标的评分规则、分值、公式、封顶及汇总，效果不合适时再讨论修改。本文保留原文件名，方便继续编辑。

## 1. 第一版的原则

**先复用 MultiRef-Compass 的评分方法，仅做开源裁判替换及必要的程序适配。不重新设计一套评分标准。**

| 项目 | 第一版做法 |
|---|---|
| 指标 | 保留原 4 个维度、14 项指标；贴图惩罚仍为 EF 辅助项 |
| 专用评分模型 | 沿用原 DOVER++、Audiobox、EF 特征模型、SyncNet、ECAPA 等计算路线 |
| 闭源裁判 | 视觉判断及清单生成改用 Qwen3.8-27B；声音及音画判断改用 Qwen3-Omni |
| 问题和量表 | 保留原检查问题、标签、档位和适用条件 |
| 算分 | 保留原公式、阈值、封顶、取整、缺失处理与已有汇总方法 |
| 分值展示 | 原来 1～5 就报 1～5，原来 0～1 就保留 0～1，PQ 等保留原值；不统一转百分制 |
| 测试集 | 使用现有题目和完整参考素材；不新出题、不补建素材、不新增整批人工标注 |
| 总分 | 先报各指标；可沿用有明确实现的 IF 组分，不自行添加 14 项统一总分 |
| 后续修改 | 先记录实际失效样例；确有问题再单独修改规则，保留旧版本供比较 |

原版依据为公开仓库固定提交 `03da121e88e241da38b41324291dfa7b8773d132`。本文按论文指标含义与该版本的实际执行代码整理；代码与提示词有差异时明确注明。部分默认提示词、缓存和权重路径未完整发布，不能据此宣称已经复现论文实验的全部配置。

**更换裁判后，即使算分函数完全相同，判断结果也可能不同。** 本方案应命名为“沿用 MultiRef-Compass 原评分规则的 Qwen 裁判配置”，不直接把成绩当成论文原 Gemini 配置的成绩。

来源：[官方指标范围](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/docs/metrics.md)、[统一运行入口](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/cli.py)。

## 2. 数据准备：直接使用现有测试题

### 2.1 每题保留的原始信息

| 信息 | 处理方式 |
|---|---|
| 样本 ID、原 prompt、已有分组 | 直接沿用，不能为补足指标而改题 |
| 参考图片、视频、音频及编号 | 使用实际发布的素材与映射；按原题指定范围读取 |
| 生成视频、待测模型及参数 | 由运行过程记录；同一道题给各模型相同输入 |
| 代码、裁判、提示词、权重版本 | 固定后保存，便于复算 |

不要求先人工填写逐题细节表、人物框、台词表、绝对时间表或关键失败条件。生成视频是待测模型对现有题目的“答题结果”，并不等于重新制作测试集。

### 2.2 自动清单与同题复用

指令遵循 IF 继续使用原自动清单流程：根据原 prompt 和已有元数据生成视觉、声音、台词与时序问题，将原闭源生成调用换为 Qwen。按原问题范围生成，不添加原题没有的要求。

例如原题为“先拿杯子，再喝水”，清单包含“拿杯子是否发生”“喝水是否发生”“拿杯子是否在喝水之前”。模型 A、B、C 的输出使用同一份问题，裁判分别读取每段视频、分别给分；不会复制另一模型的成绩。

- 清单生成时不输入任何参评模型的输出。
- 保存题目、参考内容摘要、清单、原分类、模型和模板版本；同题各模型复用。
- 题目或参考变化时生成新版；只换待测模型时复用旧清单。
- 生成与原有规则回退流程保持可追踪；接口错误可重试，仍失败时记录异常。
- 自动检查空项、重复项、参考编号和返回字段；通过格式检查不代表语义一定正确。
- **DP 使用原固定九项，DB 使用原固定六项，均不另外生成逐题细节表或绑定三元组。**
- VTF 清单保留原类别信息；不能因 Qwen 改写问句就让问题在后续分类中被静默丢弃。

来源：[原清单构建](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/build_if_eval_inputs.py)、[视觉问题范围](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/visual_checklist_generation.md)、[声音／台词／时序问题范围](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/if_checklist_generation_prompt.md)。

### 2.3 现有公开数据的覆盖范围

截至本次已核对的发布快照，Hugging Face 数据包含 350 条 prompt 及图片／部分视频参考；32 条 prompt 提及 `@Audio`，但公开包缺少对应独立音频字段、文件和映射。GitHub 的单条演示包含音频，不代表完整测试集已补齐。

第一阶段只使用**输入完整的题目**。引用了缺失音频的题目暂不纳入对应生成任务，不自行找另一段音频替代，不删掉音频要求后声称仍是原题。不能据“32 条缺音频”直接断言剩余 318 条都可用，其余素材也要自动检查。

缺音色参考并不意味着所有声音指标都不能测：其他输入完整且适用的题目，仍可测 AQ、SLS、VESA、SC、ATF 或 SCA。TS 必须具备实际需要的参考音频。

原公共输入支持多张图片，但参考视频、参考音频各为单路径。第一阶段不为多音色、多段动作或运镜参考另建题目。现有数据没有覆盖的能力，报告“本次未覆盖”，不临时补造要求。

来源：[实际 JSONL](https://huggingface.co/datasets/zxhhhhhh/MultiRef-Compass/blob/main/data/samples.jsonl)、[发布目录](https://huggingface.co/datasets/zxhhhhhh/MultiRef-Compass/tree/main)、[原输入结构](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/common/schema.py)。

## 3. 模型与开放情况

“公开可获取”与“允许任意商业使用”需要区分。当前按非商业测试用途准备，代码与权重的许可分别记录；未核实的具体派生权重不能标成已确认可用。

| 模型／组件 | 职责 | 开放情况与来源 |
|---|---|---|
| DOVER++ | VQ 技术质量 | 代码、权重公开；[代码 S-Lab License 1.0](https://github.com/VQAssessment/DOVER/blob/master/LICENSE)有非商业用途条款；[权重页](https://huggingface.co/teowu/DOVER)标 MIT，分别保留记录 |
| Qwen3.8-27B | 替换视觉裁判、清单生成及视觉辅助判断 | 公开权重，Apache-2.0；[模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Qwen3-Omni-30B-A3B-Instruct | 替换声音、音画裁判及 SLS 前筛 | 公开权重，Apache-2.0；[模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Audiobox Aesthetics | AQ 的 PQ | 公开权重，CC-BY-4.0；[模型卡](https://huggingface.co/facebook/audiobox-aesthetics) |
| ElasticFace | EF 真人脸部特征 | 官方模型公开，项目声明 CC-BY-NC-SA-4.0；[官方说明](https://github.com/fdbtrs/ElasticFace#license)。原脚本重命名权重与具体上游版本仍需对应 |
| SigLIP、DINOv2 | EF 身体／物体／场景特征及原组件诊断 | 官方权重公开，Apache-2.0；[SigLIP](https://huggingface.co/google/siglip-base-patch16-224)、[DINOv2](https://huggingface.co/facebook/dinov2-base) |
| YOLO-World 与原人脸检测组件 | EF 目标定位、裁剪 | [YOLO-World](https://github.com/AILab-CVC/YOLO-World#license)为 GPLv3；[InsightFace](https://github.com/deepinsight/insightface#license)代码 MIT，其官方模型限非商业研究。原 adapter 和具体检测权重需逐项记录 |
| LatentSync 所使用的 SyncNet 评测实现 | SLS，同步测量 | 代码与辅助权重公开；原路径为 `auxiliary/syncnet_v2.model`。[原配置](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/docs/model_weights.md)、[LatentSync 代码许可](https://github.com/bytedance/LatentSync/blob/main/LICENSE)。辅助权重条款单独记录，不按生成模型包的许可笼统处理 |
| SpeechBrain ECAPA-TDNN | TS 说话人特征与余弦相似度 | 指定权重 Apache-2.0；[模型卡](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| CurricularFace／动漫脸等原辅助资产 | 原 EF 的部分诊断或分支 | 上游项目有公开代码／模型，但原脚本所指 Glint360k 派生文件、animeface 权重等的具体来源与条款尚需逐文件对应；未核实前不宣称已完成部署 |

第一版不引入 SAM 跟踪加 Qwen 的新 EF 主评分，也不使用 Whisper 错字率代替 SCA。Qwen3.8 不承担音频判断；Qwen-Omni 必须实际接收需要的音轨。原专用模型依赖不齐时记录未就绪，不能静默换成另一个指标算法。

## 4. 分值、适用性与解释

### 4.1 各指标保留原尺度

| 指标 | 第一版报告的分值 |
|---|---|
| VQ | DOVER++ technical，0～1 |
| AQ | Audiobox PQ 原值，目标量表 1～10；不自行截断或线性换算 |
| AC | 原四维完整均值，1～5 |
| EF | 原组件、分支、LTC 和最终导出字段；不自行拼出新的唯一 EF 分 |
| DP | 原严格计分结果，1～5 |
| DB | 原三类有效分汇总，1～5 |
| SLS | 保留原脚本 `score_0_1` 与 `score_1_5`；排名显示使用原已有的 1～5 字段 |
| VESA | 原事件两轴聚合并按原代码取整，1～5 |
| SC | 原条目 1／3／5；非空条目均分，可为小数 |
| TS | 原固定阈值的 1～5 分，同时保存余弦相似度 |
| VTF | 原类别加权，1～5，可为小数 |
| ATF | 原有效项目均分向下取整，1～5 |
| TOF | 原有效项目均分向下取整，1～5 |
| SCA | 原有效台词分汇总，1～5 |

**不再统一转 0～100，不再使用我们新增的 1／0.5／0 完成率。** VQ 的 sigmoid、SLS 的 C／D 归一化和 TS 的固定阈值是原算法的一部分，保留不动；它们与新增的百分制展示不是一回事。

不同指标即使都输出 1～5，也不表示“3 分”具有相同含义。先看同一指标上的模型差异，再结合具体错误与有效评分数量解释，不能直接平均不同尺度的指标。

### 4.2 不适用、缺失和评分错误

按每个原指标的规则处理，不统一规定“缺失都是 0”“不确定都是中间分”或“错误一律跳过”。

- 原题不要求某项，与原题要求但生成失败，是不同情况。
- 数据缺参考是输入问题；评分超时是评测错误；生成结果缺内容是模型表现。三者分别记录。
- 不适用、短语音、检测失败等是否进入分母，按原具体指标执行。
- 原后端存在回退、补分或关键词触发规则时，保存实际触发原因，不只保存最后数字。
- 分数是否“通过”不在本版另定统一阈值，主要报告相对表现与明细。

## 5. 基础质量 BQ

### 01. 视觉质量 VQ

**目的与内容：** 检查清晰度、噪声、失真等技术质量。

**模型：** 原 DOVER++ 技术分支，保留原采样、预处理与片段汇总。

```text
technical = sigmoid((raw_technical − 0.1107) / 0.07355)
VQ = technical
```

保存 raw_technical 与 technical。后端已经返回 technical 时直接使用，不重复 sigmoid；不乘 100，不混入 aesthetic 或 overall。例如原结果为 0.82，报告仍为 0.82。较高通常表示技术质量较好，不代表任务遵循正确。

来源：[DOVER++ 计算](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/basic_quality/backends/legacy_bq/scripts/batch_dover.py)、[VQ 入口](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/basic_quality/visual_quality.py)。

### 02. 音频质量 AQ

**目的与内容：** 检查生成声音的制作质量，如噪声、失真与清晰度；不判断台词对不对。

**模型与计分：** Audiobox Aesthetics，直接读取原 PQ。保留原音频读取和处理方式，不新增按窗口时长加权，也不使用 `(PQ−1)/9`。例如 PQ＝7，报告就是 7。

共用基础质量流程可能同时运行视频分支，但 AQ 不融合 DOVER 分，不把其他 Audiobox 子分混入 PQ。

来源：[实际 AQ 取值](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/basic_quality/backends/legacy_bq/scripts/run_basic_quality.py#L159)、[公共 BQ 调用](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/basic_quality/_legacy_bq.py)。

### 03. 解剖／结构一致性 AC

**目的与内容：** 检查生物结构、刚体结构、接触／空间关系、场景融合四个维度。

**模型：** Qwen3.8-27B 替换原 Gemini，使用原四维提示词与量表。

**计分：** 四项各 1～5，四项都获得有效分后求算术平均。不再改成“只平均适用维度”。一项没有返回有效分时按原校验与异常流程处理，不能悄悄减小分母。例如 5、4、3、4 的 AC 为 4。

来源：[原四维提示词](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/basic_quality/backends/legacy_bq/mllm/MLLM_Prompt/anatomical%20knowledge.md)、[分数重算](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/basic_quality/backends/legacy_bq/mllm/eval_anatomical_consistency.py#L227)。

## 6. 参考保真 RF

### 04. 实体保真 EF

**目的与内容：** 比较生成的人物、物体和场景与参考是否相似，并检查跨时间保真情况。

**模型与流程：** 恢复原检测、裁剪、嵌入匹配与汇总。真人脸使用 ElasticFace，身体／物体／场景使用原 SigLIP 分支，DINOv2 等按原组件配置使用。保留原多人逐帧匹配、缺失回退与长期一致性处理，不新增“Qwen 判断身份正确后给 1／0.5／0”的算法。

原典型真人分支为 `0.6 × Face + 0.4 × Body`；原 base 的 Human／Object／Scene 权重为 0.50／0.25／0.25，并按可用分支重新归一化。这些只是计算链的一部分：人体跨帧、物体、场景及 LTC 各有原处理，不能仅靠这两行公式重写整个 EF。原场景分取有效帧相似度的最大值；原缺失实体存在 0.5 回退，也不擅自改成 0。

**输出与接入边界：** 原仓库同时提供 base、LTC 后分支，以及多种乘贴图系数后的汇总列。当前公开封装与旧导出字段存在不一致，尚不能把任一列自行称为唯一官方 EF。第一版保留各原始字段，接线时明确实际输出路径与公式；未确认的最终列记录为未就绪，不新增组合公式凑出一个总分。

例如原脚本导出的某 EF 分支为 0.8，保持 0.8；不能改成由 Qwen 判断“身份大致正确”再给另一个分数。

来源：[模型配置](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/run_entity_fidelity_from_xlsx.sh#L35)、[分支汇总](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/eval_py/run_entity_fidelity.py#L296)、[多人匹配](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/eval_py/multi_entity_fidelity.py)。

另见：[原长期一致性计算](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/tools/add_long_term_consistency.py#L25)。

### 05. 细节保持 DP

**目的与内容：** 检查参考图片中人物、物体和背景的细节是否保留。

**模型：** Qwen3.8-27B 使用原完整提示词，回答固定九项。

| 类别 | 原编号 | 检查内容概要 |
|---|---|---|
| 人物 | H1 | 发型、发色、头饰及显著人物细节 |
| 人物 | H2 | 服装类型、颜色、结构与主要布局 |
| 人物 | H3 | 纹样、标识、纽扣、围巾等局部服装和配饰 |
| 物体 | O1 | 形状、轮廓、几何及主要结构部件 |
| 物体 | O2 | 颜色、材质、纹理 |
| 物体 | O3 | 独特细节及形变、融化等异常 |
| 背景 | BG1 | 建筑、家具、门窗等关键元素 |
| 背景 | BG2 | 运镜与视角变化后，可见区域是否对应参考 |
| 背景 | BG3 | 布局、颜色光照、纹理与场景独特细节 |

固定的是问题类别，每题仍要看自己的参考。H2 可以发现“红外套变绿”，H3 可以发现“围巾条纹消失”，不需要另建逐题细节表。

**计分：** YES／PARTIAL／NO＝5／3／1；N/A 与原规则允许的 SKIP 不计入平均。使用原严格计分函数：

1. 类别内有效项先平均。有 NO 时类别上限 3；至少两项 PARTIAL 时上限 4；类别理由命中原缺失／替换关键词时上限 2。多个上限取最低。
2. 有效类别等权平均。物体类有 NO 时总分上限 4；至少两个有效类别不高于 3 时总分上限 3.5。
3. 保留最终 1～5 分，不转百分制。

例如某类 YES、YES、NO，且无更低封顶：均值约 3.67，封顶后为 3。整条视频还要按其他类别汇总，不直接把这个类别分当总分。

**实现边界：** 原代码按英文理由中的 missing、absent、replaced 等词触发上限 2，范围不只限于整类缺失。替换 Qwen 时保持机器字段与理由语言约定，并记录触发原因。完整清单优先严格重算；原解析器缺少有效清单时的子分／总分回退也需单独记录，不能误称所有结果都经过了同样的严格检查。

来源：[DP 原固定九项](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/mllm/MLLM_Prompt/detail%20preservetion.md)、[DP 严格计分代码](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/mllm/detail_preservation_eval.py#L170)。

### 06. 细节绑定 DB

**目的与内容：** 检查身份、属性与动作是否绑定到正确主体。

**模型：** Qwen3.8-27B 使用原固定六项，不生成新的绑定三元组表。

| 原类别 | 检查项 |
|---|---|
| 身份 | I2 身份是否稳定；I3 多个身份是否可区分 |
| 属性 | B1 服装归属；B3 道具归属 |
| 动作 | AC1 动作执行者；AC2 有方向的人物交互 |

**计分：** YES／PARTIAL／NO＝5／3／1，N/A 排除；先求各类别有效项目均分，再对适用类别等权平均，输出 1～5。

保留原规则：要求的绑定完全未出现，也可能标 PARTIAL；明确绑定到错误主体才按 NO 处理。例如要求 A 拿包，包完全没出现时，不能在本项自行改记 0。身份、属性、动作三个范围均保留，不只检查服装和道具。

来源：[原 DB 六项与评分规则](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/mllm/MLLM_Prompt/binding.md)。

### 辅助项：贴图惩罚

沿用原默认启用的贴图判断和系数，由 Qwen3.8-27B 替换原 Gemini。它是 EF 的辅助惩罚，不是第 15 项指标。

| 原等级 | 5 | 4 | 3 | 2 | 1 |
|---|---:|---:|---:|---:|---:|
| 对应系数 P | 1.00 | 0.95 | 0.90 | 0.80 | 0.65 |

按原出口将系数乘入相应 EF 结果；不再改为可选诊断项，也不自行放宽原贴图提示词。保留原系数解析：原 parser 可接受裁判返回的有效 penalty，缺失或无效时再按档位换算；原 shell 对缺失样本系数还可能回退模型平均值。两类来源都要记录。

示例：某原分支 0.8，实际应用系数 0.9，结果 0.72；不能后续再扣一次。EF 存在多种汇总列，必须记录本列是否已含贴图惩罚。

来源：[贴图提示词](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/mllm/MLLM_Prompt/paste%20artifact.md)、[贴图汇总](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/entity_fidelity/backends/legacy_ef/summary_py/generate_ef_paste_summary.py)。

## 7. 音画一致性 AVC

### 07. 口型同步 SLS

**目的与内容：** 判断可测说话窗口中，被选中人脸的嘴部运动与声音是否同步。

**模型：** 原 LatentSync SyncNet 评测实现与 `syncnet_v2.model`。原可选的稳定正脸说话前筛在本配置启用，Gemini 换成 Qwen3-Omni，保留原筛选要求和阈值。Qwen 不直接给同步主分。

**处理与计分：** 保留原语音窗口、人脸检测与轨迹、媒体预处理。最终排名路径从偏移绝对值小于 10 帧的候选中取最低 LSE-D 的轨迹，平手按更小绝对偏移、更高 LSE-C 排序，再保留通过稳定正脸检查的裁剪。

```text
C_norm = clip((LSE-C − 1.5) / 5.5, 0, 1)
D_norm = clip((13 − LSE-D) / 6, 0, 1)
score_0_1 = 0.60 × C_norm + 0.40 × D_norm
score_1_5 = 1 + 4 × score_0_1
```

两个分值都是原最终脚本已有字段，均保存；使用其原 1～5 排名展示，不另转百分制。同一视频内保留窗口等权平均，再按有效视频平均。例如 LSE-C＝4.25、LSE-D＝10，对应原字段 0.5 和 3。

**边界：** 原选脸可能选中同步更好的 B，不能由此证明题目要求的 A 在说话。无有效窗口时留空并报告原因，不补最低分。保留被过滤窗口与视频数量，分数只反映实际保留内容。

使用原始输出，不先修复口型或平移音轨；SyncNet 内部搜索偏移仍保留。不误读前筛分或另一套 0.8／0.2 底层分；具体出口在接入时固定。原针对个别模型与样本的历史排除名单单列，不直接套用到新模型。

来源：[最终固定映射](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/backends/legacy_ac/vsls_eval/rank_stable_frontal_lse.py)、[底层 SyncNet 实现](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/backends/legacy_ac/vsls_eval/multi_person_syncnet_eval.py)。

### 08. 视觉事件—音效匹配 VESA

**目的与内容：** 评价实际听到的非语音音效／环境声，与画面事件是否相符。

**模型：** Qwen3-Omni 使用原提示词。每个可听事件分别给声音合理性、时间匹配 1～5 分：1 表示明显错误，3 表示部分合理或对齐不稳定，5 表示合适且同步，中间档位沿用原细则。

**原代码聚合：**

```text
A = 所有有效声音合理性分的平均
T = 所有有效时间匹配分的平均
VESA = int(有效的 A、T 的平均 + 0.5)
```

时间不可定位时相应轴可无分；不是先求每个事件总分再平均，也不使用两个完成率相乘。例如两轴均为 3，总分为 3。

**边界：** 缺少 prompt 期望的音效，不在 VESA 额外扣分；没有任何可听非语音音效时不适用。原题要求的音效没生成，可由 ATF 按原规则反映。台词、口型、背景音乐不纳入本项。微弱但可辨认的音效仍可能得高分，具体沿用原档位。

来源：[原 VESA 提示词](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/backends/legacy_ac/prompts/visual_event_sound_alignment.md)、[实际后处理](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/backends/legacy_ac/eval_audio_consistency.py#L527)。

### 09. 声源正确性 SC

**目的与内容：** 判断可听语音是否能可靠归属指定可见说话人。保留原人物语音范围，不扩展到物体声源或逐角色逐句加权。

**模型：** Qwen3-Omni，使用原提示词与返回结构。

| 原条目分 | 原判断 |
|---|---|
| 1 | 有明确证据表明另一位可见人物在说话 |
| 3 | 说话人正确，但声音与画面人物的性别呈现明确不符 |
| 5 | 说话人正确，且没有上述明确不符 |
| N/A | 无声、画外音、旁白、看不清，或无法可靠归属 |

这是原提示词中的判断规则，不是对人物真实身份属性的确认。本版不重新设计该档位。仅因嘴不动或口型差，不能推断一定是错误人物；台词、语言、年龄、口音等不在本项额外扣分。

**聚合：** 原提示词要求对非空条目求算术平均，全部为空则不适用。原 parser 实际接受裁判的顶层 1～5 总分，并不自动重算上述平均；接 Qwen 时必须检查其是否遵守原格式和规则，保存不一致记录，不擅自增加取整或换算。

来源：[原 SC 提示词](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/backends/legacy_ac/prompts/source_correctness.md)。

代码细节：[条目与顶层分处理](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/backends/legacy_ac/eval_audio_consistency.py#L373)。

### 10. 音色相似度 TS

**目的与内容：** 判断生成语音与参考说话人音色的相似程度，不评台词内容。

**模型与流程：** 原 SpeechBrain ECAPA-TDNN。参考、生成音频各自经过原 energy VAD，保留片段拼接成一条音频，各提取一个向量后计算余弦相似度。保留原整段方式，不新增说话人分离或逐角色均分。

| 余弦相似度 s | 原分数 |
|---|---:|
| s ＜ 0.075 | 1 |
| 0.075 ≤ s ＜ 0.150 | 2 |
| 0.150 ≤ s ＜ 0.225 | 3 |
| 0.225 ≤ s ＜ 0.300 | 4 |
| s ≥ 0.300 | 5 |

保留原 1～5 分，同时记录余弦值及参考／生成的有效语音时长。**取消自拟 f_voice，不需要为新映射建立人工校准集。**

原入口要求两侧 VAD 保留时长均至少 2.5 秒；不足时留空并记录原因，不自动补 1。缺少路径、文件不存在、语音过短及运行异常按原状态区分。多种音色混在同一输出时，本项不提供可靠的逐角色绑定评价，第一版接受这一原方法边界。

来源：[当前 TS 公共实现](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/audio_video_consistency/voice_timbre_similarity.py)。

## 8. 指令遵循 IF

### 11. 视觉任务遵循 VTF

**目的与内容：** 检查原 prompt 中可见的动作、穿戴和其他视觉要求。

**模型：** Qwen3.8-27B，沿用原清单范围与每项 1～5 档位。

```text
VTF = 0.50 × 动作类均分 + 0.30 × 穿戴类均分 + 0.20 × 其他类均分
```

缺类别时按存在类别的权重重新归一化。最终分类脚本使用原逐项分数加权，结果可为小数。例：三类分为 5、1、1，最终为 3。

保留原边界：动作项不强制绑定指定演员，不在此评价动作先后顺序、镜头归属或参考相似度。最终实际加权类别是上述三类，不将包装层出现的其他类别名自行加权进去。

来源：[评分提示词](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/visual_task_following.md)、[动作边界](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/visual_checklist_generation.md)、[类别加权代码](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/score_visual_checklist_final.py)。

### 12. 音频任务遵循 ATF

**目的与内容：** 检查明确要求的非语音声音、音乐和环境声是否出现。

**模型：** Qwen3-Omni；原逐项 1～5 规则。

**计分：** 原公开执行代码对适用有效项先求均值，再向下取整。例如两项为 5、4，结果为 4。提示词写“均值不取整”，与实际代码有差异；本版保留已核对的代码行为并记录版本。

没有非语音要求时不适用；要求的声音缺失记 1。多余声音不因“多余”扣分，不新增禁止旁白检查；台词、音色与口型由其他指标负责。

来源：[原 ATF 提示词](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/audio_task_following.md)。

实际取整：[average_item_score](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/eval_instruction_following.py#L256)。

### 13. 时间顺序遵循 TOF

**目的与内容：** 检查可见事件的先后、必需独立镜头／场景和镜头分段结构。

**模型：** Qwen3.8-27B；每项 1～5，适用有效项均值向下取整。其提示词与代码的取整差异同 ATF，按实际执行路径记录。

保留三类原边界：

- 普通事件顺序：两个事件都可见才比较，缺任一个或无法判断时该项 N/A。
- 必需独立镜头／场景：缺失可记 1。
- 镜头分段结构：按原 1～5 细则评分。

例如“先拿杯子再喝水”只生成拿杯子，普通顺序项可 N/A，喝水是否发生由 VTF 反映。不会改成 TOF 也强制给 0。第一版不新增台词时间、声音同时发生、绝对秒数或持续时长标准。

来源：[原 TOF 规则](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/temporal_order_following.md)、[清单生成范围](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/if_checklist_generation_prompt.md)。

实际取整同：[原执行函数](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/eval_instruction_following.py#L256)。

### 14. 语音内容准确性 SCA

**目的与内容：** 检查明确指定的台词是否说对；清单只提取原题明确台词，不临时新增语义主题题目。

**模型：** Qwen3-Omni，按原大模型裁判提示词给 1～5 分。取消 Whisper＋CER／WER 主分及自定义语义完成率。

多条有效台词分先平均再向下取整，单条按其原分。原细则允许轻微同义改写；要求中文却完整正确地改用英文翻译有专门低档规则，不用一条错字比例替代。没有指定台词时不适用；要求台词但无语音时给 1；说话人是否正确不在本项扣分。

**原异常回退也要记录：** legacy 汇总对适用题没有有效评分时会补 1；缺失生成视频则单独标不适用并排除。不能把所有补 1 都解释成模型确实说错了，须保留评测错误与计分来源。

来源：[原 SCA rubric](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/if_judge_prompts/speech_accuracy.md)、[均值向下取整](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/eval_speech_content_mllm.py#L259)。

补分路径：[summary_score_from](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/eval_speech_content_mllm.py#L460)。

## 9. 最后如何评判结果

### 9.1 先比较各指标，再看原 IF 组分

各指标按原后端的单样本规则计算，再按原有效结果汇总。公共输出器对 `status=scored` 的记录求模型均值；SLS 等另有已注明的窗口到视频汇总，SCA 等原回退也需保留说明。不新增按任务类别宏平均、逐角色等权或跨指标统一尺度。

第一版默认每模型每题一次输出；需要重复实验时按轮次分开保存、报告，避免原 `sample_id＋model_name` 键覆盖结果。后续若要改变重复实验的聚合方式，再单独定规则。

原 IF 组已有明确权重，可沿用。模型级先分别计算四个指标各自的有效样本均值，再按下式合成；不改成先算每题 IF、再对题目平均：

```text
IF = 0.30 × VTF + 0.30 × ATF + 0.20 × SCA + 0.20 × TOF
```

缺维度时按可用权重重新归一化，并公开实际参与的维度和样本数；不同覆盖配置的 IF 分数不能无条件直接比较。

**取消此前自定的 BQ／RF／AVC／IF＝20%／35%／25%／20% 总分，也不把各组内部简单等权设为默认。** 原公共 CLI 没有明确的跨 14 项统一总分出口。暂以各指标表与有依据的 IF 组分报告，不把 0～1、1～5、PQ 原值直接相加。

来源：[原 IF 合成](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/instruction_following/backends/legacy_if/combine_if_all_results.py#L21)、[公共有效均值](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/common/output_writer.py)、[公共 CLI](https://github.com/zxhhh0201/MultiRef-Compass/blob/03da121e88e241da38b41324291dfa7b8773d132/finaleval/cli.py)。

### 9.2 每个分数一起展示什么

| 字段 | 用途 |
|---|---|
| 原分、分制、越高／越低越好 | 避免把不同尺度直接比较；LSE-D 等原始距离方向与主分不同 |
| 原始子项、标签、理由及回退来源 | 能查到分数是怎样算出来的 |
| 有效评分数、未评分数与原因 | 防止只看较高均分，却忽略大量样本没测成 |
| 生成失败与评测错误 | 区分模型没有生成，与裁判或程序没有完成 |
| 代码、模型、权重、提示词与数据版本 | 复算与后续改规则时进行对照 |

公共 `num_skipped` 包含所有非 scored，可能与 `num_failed` 重叠；保留原统计时说明口径，不把它们相加成互斥总数。

同一指标上分数更高、有效覆盖相近，才较有依据地说该模型在该项更好。结果差异很小时查看具体样例及重复性，不新增一个未经验证的“及格线”作结论。缺失题、N/A 和错误率另列，不通过新增惩罚混进原分数。

## 10. 必要适配与验证

### 10.1 需要修改的程序

1. **替换闭源调用。** 包括最终裁判、自动清单、贴图判断、SLS 前筛，以及其他实际调用 Gemini 的辅助步骤。保留原问题与 JSON 字段。
2. **正确传媒体。** 图片、视频、音轨按原用途送入 Qwen；替换接口不等于已经证明裁判看到或听到了完整输入。
3. **对齐公开文件。** 默认 `if_judge_prompts_v3` 与预存清单有未发布文件，改用实际可获取的提示词和生成流程，记录版本差异。
4. **保留 VTF 分类。** 原按全文查类别、查不到 DROP。新清单必须带可追踪的原类别映射，不能无声丢题。
5. **接对 SLS／EF 结果。** 修复文件名与字段不一致，区分原始分、前筛分、LTC 后分及贴图后分；不借接线之名重写公式。
6. **检查历史特例。** 针对旧模型名字的跳过名单、特定样本排除等显式配置，不直接继承为新模型的评分规律。
7. **固定模型资产。** 完成具体权重来源、依赖与本次用途条款记录；不替换成来源不明的同名文件。

这些是接入与运行工作，不要求重新制作测试集。所列问题来自代码核查，尚未通过本地运行确认已修复。

### 10.2 先验证能否按原规则运行

用少量输入完整的现有题目，检查媒体送入、返回格式、算分出口和可复算性。对固定标签手工核对少量已知结果，例如 DP 的封顶、TS 的阈值边界、ATF／TOF 的向下取整。

可在已有输出上做少量排错，例如延迟声音观察 SLS 的分数与保留窗口数如何变化；原规则可能把严重不同步窗口排除，因此不能只期待总分一定下降。检查重复评分是否波动、Qwen 是否经常忽略某类证据。

无需为启动建立整批人工校准集。人工只处理异常，少量抽查可选；没有验证与人类评价一致前，结果用于探索性比较，不声称已充分证明排名可靠。

### 10.3 效果不好时再改什么

先保存具体失效样例，区分是媒体传错、程序错误、Qwen 判断错误，还是原评分规则本身不适合。前两类先修实现；需要改裁判提示词、缺失惩罚、权重或指标范围时，再提出明确的新版本。

新规则不覆盖旧基线，所有参评模型在同一规则下统一重评。本文以前提出的多角色逐句同步、动态细节表、新音色校准、CER／WER 主分、额外漏音效扣分等，均不作为当前默认任务。

## 11. 当前决定与尚未完成的工作

已确认：

- [x] 使用现有完整测试题，不重建测试集或增加整批人工标注。
- [x] VQ 使用原 DOVER++ technical。
- [x] 全部指标优先沿用原公开实现的评分规则、分值与汇总；取消统一百分制和自定义总分。
- [x] DP 原九项、DB 原六项；SLS 原同步程序；TS 原整段特征与固定阈值。
- [x] 原闭源裁判替换为本地 Qwen／Qwen-Omni，原专用模型路线保持。
- [x] 原方法的边界先如实报告，后续根据实测再决定是否改规则。

待完成的是实现与验证，不是重新选择一套分值：

- [ ] 补齐可获取的原模型资产，确认具体文件来源与许可。
- [ ] 核对 EF 最终导出列和 SLS 固定评分出口，修复接线。
- [ ] 接入 Qwen、缓存清单并核对类别映射。
- [ ] 在现有完整题目上跑通少量端到端评测，报告覆盖和错误。
- [ ] 记录代码与提示词矛盾，以及裁判替换后的判断差异。

### 讨论记录

| 日期 | 讨论 | 当前决定 |
|---|---|---|
| 2026-10-23 | 不商用；VQ 选型 | 使用 DOVER++ technical，许可按代码与权重分别记录 |
| 2026-10-23 | 自行标注工作量过大 | 沿用现有数据与原自动清单，不做全量人工标注 |
| 2026-10-23 | 不考虑重建测试集 | 新题、补建素材、额外整批校准数据暂缓 |
| 2026-10-23 | SLS 逐人物逐句方案太复杂 | 使用原同步程序、选脸与固定公式 |
| 2026-10-23 | DP 动态细节表没有必要 | 使用原九项及封顶算法 |
| 2026-10-23 | 百分制是否为原版 | 取消新增百分制，保留原已有分值 |
| 2026-10-23 | 所有分值方法先按原版，后续效果不好再改 | 本文全面移除此前自定义计分与汇总；默认原规则，只做必要开源适配。尚未实现或运行评测。 |


