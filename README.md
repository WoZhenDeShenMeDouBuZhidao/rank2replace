# Rank2Replace

```text
rank2replace/
  setup.sh               環境建置與模型下載
  requirements.txt       固定套件版本
  .venv/                 訓練、demo、vLLM 共用環境
  reproduction/          主實驗一至三的資料、訓練與作圖
  vision_pipeline_demo/  Extract + Rank、Environment annotation
  verification.json      已完成的驗證紀錄
```

## 執行

資料與 demo 範例圖片目前未隨 repository 提供，下載方式待補；訓練與 demo 推論需要先備妥對應資料。

需要 Linux、Python 3.10，以及相容的 NVIDIA driver。從本目錄建置環境並下載 demo 模型：

```bash
bash setup.sh --download-models
```

模型存入 `vision_pipeline_demo/models/`，已有完整檔案會重用。

接著依 [主實驗復現](reproduction/README.md) 或 [影像前處理 demo](vision_pipeline_demo/README.md) 執行。
