# train.py
# ---------------------------------------------------------
# 這是專門用來 "訓練" 的腳本
# 執行方式：在 Terminal 輸入 python train.py
# ---------------------------------------------------------

import os
import multiprocessing
import torch
from pathlib import Path

# 1. Windows 必要防崩潰設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 匯入 Anomalib
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from lightning.pytorch import seed_everything 

def main():
    print("🔒 設定 Random Seed: 42")
    seed_everything(100, workers=True)
    print("=== 1. 硬體與環境檢查 ===")
    if torch.cuda.is_available():
        print(f"✅ GPU 準備就緒: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: 未偵測到 GPU，將使用 CPU 訓練")

    # 設定資料路徑
    dataset_root = Path("./datasets")

    print("=== 2. 建立資料集 (DataModule) ===")
    # 這裡只負責定義資料在哪裡
    datamodule = Folder(
        name="toothbrush",
        root=dataset_root,
        #normal_dir="train/good",
        normal_dir="train/multi_good",
        # abnormal_dir="test/defect",
        # normal_test_dir="test/good",
        # 批次大小 (小一點比較穩定)
        train_batch_size=8,
        eval_batch_size=8,
        # Windows 必填 0
        num_workers=0
    )

    try:
        datamodule.setup()
        print("✅ DataModule 設定成功")
    except Exception as e:
        print(f"❌ DataModule 設定失敗: {e}")
        return

    print("=== 3. 建立模型 (Model) ===")
    model = Patchcore(
        backbone="wide_resnet50_2",       
        #layers = ["layer2", "layer3"],
        coreset_sampling_ratio=1, 
    )
    print(f"🧐 目前使用的 Layers: {model.hparams.layers}")
    print("=== 4. 初始化引擎 (Engine) ===")
    engine = Engine(
        accelerator="auto",            
        devices=1,
        default_root_dir="./results", # 模型與結果會存到這裡
        # 注意：不再需要 task 參數
    )

    print("=== 5. 開始訓練 (Fit) ===")
    try:
        # 開始訓練
        engine.fit(datamodule=datamodule, model=model)
        print("🎉🎉🎉 訓練成功完成！ 🎉🎉🎉")
        
        # 訓練完馬上測試一下，計算 AUROC 分數
        print("=== 6. 執行測試 (Test) ===")
        engine.test(datamodule=datamodule, model=model)
        
        print("\n✅ 流程結束。模型已儲存至 results/ 資料夾。")
        print("現在請開啟 .ipynb 進行視覺化查看。")

    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windows 多工處理保護
    multiprocessing.freeze_support()
    main()
