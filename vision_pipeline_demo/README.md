# 影像前處理 Demo

```text
vision_pipeline_demo/
  extract_rank.py          FT-DINOSAUR extraction + Qwen ranking
  annotate_environment.py  原圖的 environment annotation
  prompts.py              各 dataset 的 prompt 與 environment 選項
  ranking_prompts.md      Score、Binary、Sequential 的 prompt 說明
  common.py               共用圖片讀取與 API client
  serve_qwen.sh           啟動 Qwen vLLM API
  download_models.py      下載固定版本模型
  models/                 FT-DINOSAUR、Qwen3-VL-8B-Instruct
  assets/                 固定 slot 初始化
  vendor/                 FT-DINOSAUR 實作
  outputs/
    environment/
      original_results/   原圖與我們保存的 environment labels
      reproduced_results/ 重新標註的結果
    extract_rank/
      original_results/   原圖與我們保存的 slots、ranking、comparison.png
      reproduced_results/ 重新 extraction／ranking 的結果
```

## 執行

先完成[根目錄環境建置](../README.md#執行)。在第一個終端啟動 API，將 key 換成自己的私密值：

```bash
cd vision_pipeline_demo
export VLLM_API_KEY='YOUR_PRIVATE_API_KEY'
CUDA_VISIBLE_DEVICES=0 bash serve_qwen.sh
```

API 使用共用 `.venv`，僅監聽 `127.0.0.1:8000`。第二個終端設定相同 key，再分別執行兩個獨立流程：

```bash
cd vision_pipeline_demo
export VLLM_API_KEY='YOUR_PRIVATE_API_KEY'

# CPU extraction，再呼叫 Qwen 排名
../.venv/bin/python extract_rank.py --device cpu

# 原圖直接送 Qwen 標環境，不經過 extraction 或 ranking
../.venv/bin/python annotate_environment.py
```

每個流程各有八張示範圖片；可加 `--samples waterbirds_000001` 只跑一張。`000001` 是原始 metadata 從 0 起算的 row index。結果存入各自的 `reproduced_results/`，同張圖片再次執行會覆寫；用 `--output` 可指定另一個位置。`ranking.json` 只保存 Direct Rank 的 `rank_order`，由重要到不重要列出原始 slot index。API 使用完後在第一個終端按 Ctrl+C 停止。

每次執行都重新推論，不讀取保存的排名或標註作為答案。Direct Rank 統一使用共享 template。
