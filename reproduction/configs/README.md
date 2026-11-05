# 實驗設定

```text
configs/
  experiment1.json  Overall performance
  experiment2.json  原始 slots、Qwen 排序、排序加 corruption
  experiment3.json  Oracle、noise 與各 VLM 的 environment labels
```

## 執行

每份 JSON 的 `jobs` 已展開所有 data splits、超參數組合與三個 seeds（9803、38224、8113）。在 `reproduction/` 執行對應 script：

```bash
# N 換成 1、2 或 3
bash scripts/experimentN.sh --dry-run
bash scripts/experimentN.sh --gpus 0,1 --workers-per-gpu 4
```

超參數直接以 JSON 為準；訓練與報告指令見[上層 README](../README.md)。
