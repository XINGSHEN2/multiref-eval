# Detail Preservation 严格评分说明

当前 `detail_preservation_eval.py` 默认使用严格 checklist item 评分，并且标准输出只保留这一版排名。

## 当前子项

### Human Detail

- `H1`: 发型、发色、头饰/头部细节，以及妆容、胡子、纹身、疤痕、雀斑、痣、徽章、身体轮廓、角色特有身体结构等 distinctive human details。旧版 `H1` 和 `H4` 已合并为这一项。
- `H2`: 衣服类型、衣服颜色、服装结构、主要服装布局。
- `H3`: 局部服饰/配饰细节，例如图案、logo、纽扣、领口、腰带、首饰、眼镜、包、围巾、手套、手持物。

### Object Detail

- `O1`: 物体形状、轮廓、几何结构、主要结构部件。
- `O2`: 物体颜色、颜色布局、材质、纹理。
- `O3`: 物体独特细节和稳定性，例如 logo、标签、图案、鞋带、把手、接缝、装饰件，以及是否有融化、形变、非预期转变。

### Background Detail

- `BG1`: 关键背景元素。
- `BG2`: 镜头运动、裁切或视角变化下，背景区域是否仍匹配参考图。
- `BG3`: 空间布局、颜色/光照、材质纹理、标志性背景细节。

## Item 分数

- `YES = 5`
- `PARTIAL = 3`
- `NO = 1`
- `N/A` 和 `SKIP` 不计分

## Section 分数

每个 section 先对可计分 item 取平均：

```text
section_raw_score = mean(non-NA, non-SKIP item scores)
```

然后应用 section cap：

- 任意 item 为 `NO`，section 最高为 `3`
- 两个或以上 `PARTIAL`，section 最高为 `4`
- section reason 显示主参考人/物/背景缺失、被替换或完全丢失，section 最高为 `2`

```text
section_score = min(section_raw_score, section_cap)
```

## 总分

总分先对所有适用 section 的最终 section score 做算术平均：

```text
detail_score_raw = mean(applicable section scores)
```

然后应用 total cap：

- `object_detail` 中任意 item 为 `NO`，总分最高为 `4`
- 两个或以上 section 的分数 `<= 3`，总分最高为 `3.5`

```text
Detail Preservation Score = min(detail_score_raw, total_cap)
```

## 输出

以后标准输出文件就是这一版严格评分：

- `detail_preservation_per_sample.csv`
- `detail_preservation_item_detail.csv`
- `detail_preservation_model_summary.csv`
- `model_ranking_overall_total.csv`
- `model_ranking_overall_dimensions.csv`
- `model_ranking_by_board_total.csv`
- `model_ranking_by_board_dimensions.csv`
- `model_rankings.md`
