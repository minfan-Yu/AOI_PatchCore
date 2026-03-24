import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Sentech 相機支援 ---
import stapipy as st
import platform

# --- Anomalib 相關引用 ---
from anomalib.models import Patchcore

# ==========================================
# [區域 A] 系統與模型設定
# ==========================================
USE_SENTECH_CAMERA = True       # <<< 設為 True 使用 Sentech 相機
VIDEO_PATH = 'test_long.mp4'    # <<< 若 USE_SENTECH_CAMERA=False 則用這個
SAVE_DIR = './detected'
BG_SAVE_DIR = './background'
CKPT_PATH = "model.ckpt"

# [重要] 閾值設定
THRESHOLD = 50                 

# ==========================================
# [區域 B] 影像處理參數
# ==========================================
ROI_RECT = (1500, 800, 1000, 1300) # (x, y, w, h)
DIFF_THRESHOLD = 30
MIN_AREA_THRESHOLD = 1000
MOVEMENT_THRESHOLD = 800
STABLE_CONSECUTIVE_FRAMES = 10

# ==========================================
# [Sentech 相機初始化]
# ==========================================
def configure_sentech_env():
    current_os = platform.system()
    if current_os == "Linux":
        os.environ["GENICAM_GENTL64_PATH"] = "/opt/sentech/lib"
    elif current_os == "Windows":
        gentl_path = r"C:\Program Files\Common Files\OMRON_SENTECH\GenTL\v1_5"
        if os.path.exists(gentl_path):
            os.environ["GENICAM_GENTL64_PATH"] = gentl_path
            print(f"[*] Sentech 驅動路徑: {gentl_path}")

def init_sentech_camera():
    configure_sentech_env()
    st.initialize()
    st_system = st.create_system()
    st_device = st_system.create_first_device()
    print(f"[*] 成功連線相機: {st_device.info.display_name}")

    # ✅ [優化] 建立 SDK 內建 SIMD 加速轉換器，取代主迴圈中的 cv2.cvtColor
    st_converter = st.create_converter(st.EStConverterType.PixelFormat)
    st_converter.destination_pixel_format = st.EStPixelFormatNamingConvention.BGR8
    print("[*] 已啟動 SDK SIMD 加速轉換器 (Bayer → BGR8)")

    st_datastream = st_device.create_datastream()
    st_datastream.start_acquisition()
    st_device.acquisition_start()

    # ✅ 將轉換器一併回傳，供主迴圈使用
    return st_device, st_datastream, st_converter

# ==========================================
# [步驟 1] 載入模型
# ==========================================
print(">>> 正在載入 PatchCore 模型，請稍候... <<<")
from glob import glob
ckpt_files = glob(CKPT_PATH, recursive=True)
if not ckpt_files:
    raise FileNotFoundError("找不到 .ckpt 檔案，請確認路徑是否正確。")

lightning_model = Patchcore.load_from_checkpoint(ckpt_files[-1], map_location=torch.device('cpu'))
lightning_model.eval()
model = lightning_model.model.cpu()
print("✅ 模型已載入至 CPU")

# ==========================================
# [核心功能] 快速推論與繪圖
# ==========================================
def infer_anomaly(model, roi_image):
    start_time = time.time()
    
    # Preprocess
    img_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0).cpu()

    with torch.inference_mode():  # ✅ [優化] 比 no_grad 更快，省掉 autograd 版本追蹤
        outputs = model(img_tensor)
    
    anomaly_map = None
    pred_score = None

    anomaly_map = outputs[2].squeeze().cpu().numpy()
    pred_score = outputs[0].item()

    if anomaly_map is None: raise ValueError("無法從模型輸出中找到 Anomaly Map")
    if pred_score is None: pred_score = float(anomaly_map.max())

    elapsed = time.time() - start_time
    print(f"⏱️  推論時間: {elapsed*1000:.1f}ms | Score: {pred_score:.4f}")
    return anomaly_map, pred_score

def save_anomaly_plot(img_roi, anomaly_map, raw_score, threshold, save_path):
    try:
        img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
        img_vis = cv2.resize(img_rgb, (256, 256)) 
        if anomaly_map.max() - anomaly_map.min() > 1e-6:
            map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            map_normalized = np.zeros_like(anomaly_map)
            
        is_ng = raw_score > threshold
        status_str = "NG" if is_ng else "OK"
        img_draw = img_vis.copy()
        
        if is_ng:
            mask = ((anomaly_map > threshold) * 255).astype(np.uint8)
            if np.sum(mask) == 0:
                local_max = anomaly_map.max()
                mask = ((anomaly_map > local_max * 0.90) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > 10: cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_vis); axes[0].set_title('Original'); axes[0].axis('off')
        im = axes[1].imshow(map_normalized, cmap='jet', vmin=0, vmax=1); axes[1].set_title('Heatmap'); axes[1].axis('off')
        axes[2].imshow(img_draw); axes[2].set_title(f'Result: {status_str}', color='red' if is_ng else 'green'); axes[2].axis('off')
        plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✅ 報告已儲存: {save_path}")
    except Exception as e:
        print(f"❌ 繪圖存檔失敗: {e}")

# ✅ [優化] 非同步存圖：把 matplotlib 存圖丟到背景 thread，不阻塞主迴圈
# 用 Semaphore 限制同時最多 2 個存圖 thread，避免記憶體爆炸
_save_semaphore = threading.Semaphore(2)

# ✅ [優化] 非同步推論：用 Lock + flag 確保同時只有一個推論 thread 在跑
_infer_lock = threading.Lock()
_infer_result = {"anomaly_map": None, "raw_score": None, "status": "idle"}
# status: "idle" | "running" | "done"

def save_anomaly_plot_async(img_roi, anomaly_map, raw_score, threshold, save_path):
    """非同步版本：在背景 thread 執行存圖，主迴圈立即繼續"""
    def _worker():
        with _save_semaphore:  # 同時最多 2 個存圖 thread
            save_anomaly_plot(img_roi, anomaly_map, raw_score, threshold, save_path)
    
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t  # 回傳 thread 物件，需要時可以 join

def infer_anomaly_async(model, roi_image, timestamp):
    """
    ✅ [優化] 推論 + 存圖在同一個背景 thread 裡依序執行：
    避免存圖和推論同時搶 CPU 導致推論變慢。
    主迴圈完全不等待。
    """
    def _worker():
        if not _infer_lock.acquire(blocking=False):
            print("[!] 上一次推論尚未完成，略過此次觸發")
            return
        try:
            _infer_result["status"] = "running"

            # Step 1: 推論（CPU 密集）
            anomaly_map, raw_score = infer_anomaly(model, roi_image)

            # Step 2: 推論完才存圖，不互搶 CPU
            is_ng = raw_score > THRESHOLD
            status_str = "NG" if is_ng else "OK"
            print(f">>> [分析完成] Raw Score: {raw_score:.4f} | 判定: {status_str}")
            save_filename = f"{SAVE_DIR}/report_{timestamp}_{status_str}.png"
            save_anomaly_plot(roi_image, anomaly_map, raw_score, THRESHOLD, save_filename)

            # Step 3: 結果寫回讓主迴圈更新畫面文字
            _infer_result["raw_score"] = raw_score
            _infer_result["status_str"] = status_str
            _infer_result["status"] = "done"

        except Exception as e:
            print(f"推論錯誤: {e}")
            _infer_result["status"] = "idle"
        finally:
            _infer_lock.release()

    threading.Thread(target=_worker, daemon=True).start()

# ==========================================
# [步驟 2] 主程式迴圈
# ==========================================
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if not os.path.exists(BG_SAVE_DIR): os.makedirs(BG_SAVE_DIR)

st_device = None; st_datastream = None; st_converter = None

if USE_SENTECH_CAMERA:
    print(">>> 使用 Sentech 相機模式 <<<")
    # ✅ 接收三個回傳值（新增 st_converter）
    st_device, st_datastream, st_converter = init_sentech_camera()
else:
    print(f">>> 使用影片檔案: {VIDEO_PATH} <<<")
    cap = cv2.VideoCapture(VIDEO_PATH)

prev_gray = None
avg_float = None  
stable_counter = 0
trigger_cooldown = 0
state = "INIT"
last_result_text = "Ready..."
last_result_color = (255, 255, 255)
last_result_time = 0
print(f">>> 系統啟動完成，背景將鎖定不更新 <<<")

try:
    while True:
        if USE_SENTECH_CAMERA:
            with st_datastream.retrieve_buffer(5000) as st_buffer:
                if not st_buffer.info.is_image_present: continue
                
                # ✅ [優化] 使用 SDK 轉換器執行 Bayer → BGR，取代原本兩步驟：
                #    原本: np.frombuffer → reshape(H,W) → cv2.cvtColor(BayerBG2BGR)
                #    現在: SDK 直接轉換，CPU 負擔更低
                raw_image = st_buffer.get_image()
                st_converted = st_converter.convert(raw_image)
                data = st_converted.get_image_data()
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    st_converted.height, st_converted.width, 3)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # 1. 裁切 ROI
        x, y, w, h = ROI_RECT
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # 背景初始化與存檔
        if avg_float is None:
            print(">>> [系統] 捕捉初始背景 (Background Locked) <<<")
            avg_float = np.float32(gray_blur) 
            prev_gray = gray_blur.copy()
            bg_path = f"{BG_SAVE_DIR}/background.png"
            cv2.imwrite(bg_path, roi)
            print(f">>> [系統] 背景已儲存至: {bg_path}")
            continue

        bg_frame = cv2.convertScaleAbs(avg_float)

        # 2. 動態與存在偵測
        diff_motion = cv2.absdiff(gray_blur, prev_gray)
        _, thresh_motion = cv2.threshold(diff_motion, 25, 255, cv2.THRESH_BINARY)
        motion_score = cv2.countNonZero(thresh_motion)

        diff_presence = cv2.absdiff(gray_blur, bg_frame)
        _, thresh_presence = cv2.threshold(diff_presence, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        presence_area = cv2.countNonZero(thresh_presence)

        # 3. 狀態機判定
        if trigger_cooldown > 0:
            trigger_cooldown -= 1
            state = f"COOLDOWN ({trigger_cooldown})"
            color = (0, 0, 255)
        else:
            if motion_score < MOVEMENT_THRESHOLD and presence_area > MIN_AREA_THRESHOLD:
                stable_counter += 1
                state = f"LOCKED: {stable_counter}"
                color = (0, 255, 255)
                
                if stable_counter >= STABLE_CONSECUTIVE_FRAMES:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    # ✅ [優化] 非同步推論：立即返回，不卡主迴圈
                    infer_anomaly_async(model, roi.copy(), timestamp)
                    state = "ANALYZING..."
                    trigger_cooldown = 45
                    stable_counter = 0
            else:
                stable_counter = 0
                if presence_area <= MIN_AREA_THRESHOLD:
                    state = "WAITING"
                    color = (100, 100, 100)
                else:
                    state = f"MOVING"
                    color = (255, 0, 0)

        # ✅ [優化] 每幀檢查背景推論是否完成，完成就取結果並觸發存圖
        if _infer_result["status"] == "done":
            raw_score  = _infer_result["raw_score"]
            status_str = _infer_result["status_str"]
            _infer_result["status"] = "idle"  # 重設，避免重複處理

            last_result_color = (0, 0, 255) if status_str == "NG" else (0, 255, 0)
            last_result_text  = f"Result: {status_str} ({raw_score:.2f})"
            last_result_time  = time.time()

        prev_gray = gray_blur.copy()

        if last_result_time != 0 and (time.time() - last_result_time > 10.0):
            last_result_text = "Ready..."
            last_result_color = (255, 255, 255)
            last_result_time = 0

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 20)
        cv2.putText(frame, state, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)
        cv2.putText(frame, last_result_text, (x, y+h+100), cv2.FONT_HERSHEY_SIMPLEX, 3, last_result_color, 10)
        
        display_frame = cv2.resize(frame, (1024, 768))
        cv2.imshow('Smart Inspection System', display_frame)
        
        key = cv2.waitKey(30 if not USE_SENTECH_CAMERA else 1)
        if key == 27: break
        elif key == ord('r'):
            avg_float = np.float32(gray_blur) 
            bg_path = f"{BG_SAVE_DIR}/background.png"
            cv2.imwrite(bg_path, roi)
            print(f">>> [指令] 背景已更新並儲存至: {bg_path} <<<")

finally:
    if USE_SENTECH_CAMERA:
        if st_device: st_device.acquisition_stop()
        if st_datastream: st_datastream.stop_acquisition()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("[*] 系統已安全關閉")
