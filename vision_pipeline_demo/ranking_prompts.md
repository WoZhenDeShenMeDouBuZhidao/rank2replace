# 其他 Ranking Prompts

Demo 只執行 Direct Rank。以下保留另外三種方法的原實驗 prompt；`{task}` 使用 `prompts.py` 中 `DATASET_CONFIG` 的 `ranking_task`。空白 slots 不送入 VLM，最後補到排名尾端。

## Pointwise Score

每次附一張非空 slot image：

```text
Task: {task}. Rate this masked region from 0 to 9. Gray pixels are excluded. Higher means stronger intrinsic evidence for identifying the target class, robust to environment changes; 0 means no useful evidence, 9 means very strong evidence. Judge only visible pixels. Return exactly one digit, without explanation.
```

Constrained decoding 只允許 `0`～`9`，取得十個 token 的 log probabilities，正規化後算期望分數，再由高到低排序。

## Binary Selection

每次附一張非空 slot image：

```text
Task: {task}. Should this masked region be selected as useful intrinsic evidence for identifying the target class, robust to environment changes? Gray pixels are excluded. Judge only visible pixels. Return exactly Yes or No, without explanation.
```

Constrained decoding 只允許 `Yes`／`No`，正規化兩個 token 的機率後，依 `P(Yes)` 由高到低排序。Pointwise Score 與 Binary Selection 同分時依原 slot ID 遞增。

## Sequential Selection

沿用 `prompts.py` 中 Direct Rank 的共享 template，將最後從 `Return ...` 開始的句子替換為：

```text
Select the single most important region among these remaining IDs: {remaining_ids}. Return only its ID, without explanation.
```

`{remaining_ids}` 例如 `S0,S2,S5`。每輪只附剩餘非空 slot images，依原 slot ID 遞增展示，constrained decoding 只允許其中一個 ID。選中的 slot 加到排名尾端並移出候選，最後一個直接補入；不帶入前輪對話。
