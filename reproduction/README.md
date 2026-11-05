# 主實驗一至三復現

```text
reproduction/
  scripts/experiment{1,2,3}.sh   三組實驗入口
  configs/experiment{1,2,3}.json 完整 configurations 與超參數
  rank2replace/                訓練、選模、統計與作圖
  data/<dataset>/
    images/                    原始圖片
    metadata.csv               圖片 row index
    slot_images/               Slot PNG 的 tar 分片
    slot_embeddings/           原始 slots 與 masks
    ranking/                   Qwen ranking
    environment_inference/     VLM environment labels
    splits/                    固定切分、標準化輸入與 training groups
  outputs/exp{1,2,3}/
    original_results/
      results.json             原實驗 metrics
      summary.json, sweep.csv  選模與彙整結果
      *.pdf                    原實驗圖表
    reproduced_results/
      runs/                    重新訓練的 checkpoints、metrics 與 logs
      smoke/                   Smoke test 結果
      summary.json, sweep.csv  重新選模與彙整結果
      *.pdf                    重新訓練的圖表
  implementation_sources.json  第三方實作來源
```

## 執行

先完成[根目錄環境建置](../README.md#執行)，再從根目錄執行：

```bash
cd reproduction

# 查看實驗規模
bash scripts/experiment1.sh --dry-run

# CPU smoke test
bash scripts/experiment1.sh --device cpu --smoke --datasets metashift --methods ce --limit 1

# 實驗一：Overall performance
bash scripts/experiment1.sh --gpus 0,1 --workers-per-gpu 4

# 實驗二：原始 slots、排序、排序加 corruption
bash scripts/experiment2.sh --gpus 0,1 --workers-per-gpu 4

# 實驗三：Environment label 品質
bash scripts/experiment3.sh --gpus 0,1 --workers-per-gpu 4
```

依可用 GPU 調整 `--gpus` 與 `--workers-per-gpu`。可用 `--datasets`、`--methods`、`--arms`、`--qualities` 篩選；已完成的 run 會跳過。各 data split／方法／條件依三個 seeds 的平均 validation selection 選超參數。

```bash
# 查看進度與產生重新訓練的圖表；N 換成 1、2 或 3
../.venv/bin/python -m rank2replace.progress --experiment N
../.venv/bin/python -m rank2replace.report --experiment N

# 只重畫原實驗結果，不重新訓練
../.venv/bin/python -m rank2replace.report --experiment N --reference
```

結果與圖表一起存入 `outputs/expN/original_results/` 或 `outputs/expN/reproduced_results/`。`R2R_OUTPUT` 可更改新結果的輸出根目錄，仍依 `expN/reproduced_results/` 分組。實驗三的圖另需實驗一的 Rank2Replace 結果。
