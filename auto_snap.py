import cv2
import numpy as np
import os
import time
from datetime import datetime
import platform

# --- Sentech 相機支援 ---
# 嘗試匯入，若無安裝 SDK 則自動略過
try:
    import stapipy as st
    SENTECH_AVAILABLE = True
except ImportError:
    SENTECH_AVAILABLE = False

# ==========================================
# [區域 A] 系統設定
# ==========================================
USE_SENTECH_CAMERA = True       # <<< 設為 True 使用 Sentech 相機
VIDEO_PATH = 'test_long.mp4'    # <<< 若 USE_SENTECH_CAMERA=False 則用這個
SAVE_DIR = './images'    # 🔥 圖片蒐集存放路徑
BG_SAVE_DIR = './background'    # 背景儲存路徑

# ==========================================
# [區域 B] 影像處理參數 (保持與 main.py 一致)
# ==========================================
ROI_RECT = (1500, 800, 1000, 1300) # (x, y, w, h)
DIFF_THRESHOLD = 30
MIN_AREA_THRESHOLD = 1000
MOVEMENT_THRESHOLD = 800
STABLE_CONSECUTIVE_FRAMES = 10     # 連續靜止幾幀才截圖
RESULT_RESET_SECONDS = 5.0         # 顯示結果幾秒後重置文字

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

def init_sentech_camera():
    if not SENTECH_AVAILABLE:
        print("❌ 未安裝 stapipy，無法使用 Sentech 相機")
        return None, None
    configure_sentech_env()
    st.initialize()
    st_system = st.create_system()
    st_device = st_system.create_first_device()
    print(f"[*] 成功連線相機: {st_device.info.display_name}")
    st_datastream = st_device.create_datastream()
    st_datastream.start_acquisition()
    st_device.acquisition_start()
    return st_device, st_datastream

# ==========================================
# [主程式迴圈]
# ==========================================
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if not os.path.exists(BG_SAVE_DIR): os.makedirs(BG_SAVE_DIR)

st_device = None; st_datastream = None
cap = None

if USE_SENTECH_CAMERA and SENTECH_AVAILABLE:
    print(">>> 使用 Sentech 相機模式 (蒐集資料用) <<<")
    st_device, st_datastream = init_sentech_camera()
else:
    print(f">>> 使用影片/Webcam 模式: {VIDEO_PATH} <<<")
    cap = cv2.VideoCapture(VIDEO_PATH)

prev_gray = None
avg_float = None  
stable_counter = 0
trigger_cooldown = 0
saved_count = 0

# UI 狀態變數
state = "INIT"
last_result_text = "Ready..."
last_result_color = (255, 255, 255)
last_result_time = 0

print(f">>> 系統啟動完成，請將物體放入框中以自動截圖 <<<")
print(f">>> 圖片將儲存至: {SAVE_DIR}")

try:
    while True:
        # 1. 取得影像
        if USE_SENTECH_CAMERA and st_datastream:
            with st_datastream.retrieve_buffer(5000) as st_buffer:
                if not st_buffer.info.is_image_present: continue
                st_image = st_buffer.get_image()
                data = st_image.get_image_data()
                raw_frame = np.frombuffer(data, dtype=np.uint8).reshape(st_image.height, st_image.width)
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_BayerBG2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # 2. 裁切 ROI
        x, y, w, h = ROI_RECT
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # 3. 背景初始化與存檔
        if avg_float is None:
            print(">>> [系統] 捕捉初始背景 (Background Locked) <<<")
            avg_float = np.float32(gray_blur) 
            prev_gray = gray_blur.copy()
            
            # 儲存背景圖片
            bg_path = f"{BG_SAVE_DIR}/background.png"
            cv2.imwrite(bg_path, roi)
            print(f">>> [系統] 背景已儲存至: {bg_path}")
            continue

        bg_frame = cv2.convertScaleAbs(avg_float)

        # 4. 動態與存在偵測
        diff_motion = cv2.absdiff(gray_blur, prev_gray)
        _, thresh_motion = cv2.threshold(diff_motion, 25, 255, cv2.THRESH_BINARY)
        motion_score = cv2.countNonZero(thresh_motion)

        diff_presence = cv2.absdiff(gray_blur, bg_frame)
        _, thresh_presence = cv2.threshold(diff_presence, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        presence_area = cv2.countNonZero(thresh_presence)

        # 5. 狀態機判定 (資料蒐集邏輯)
        if trigger_cooldown > 0:
            trigger_cooldown -= 1
            state = f"COOLDOWN ({trigger_cooldown})"
            color = (0, 0, 255) # 紅色
        else:
            if motion_score < MOVEMENT_THRESHOLD and presence_area > MIN_AREA_THRESHOLD:
                stable_counter += 1
                state = f"LOCKED: {stable_counter}"
                color = (0, 255, 255) # 黃色
                
                if stable_counter >= STABLE_CONSECUTIVE_FRAMES:
                    # === 觸發截圖 ===
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    save_path = f"{SAVE_DIR}/train_{timestamp}.png"
                    
                    # 儲存 ROI
                    cv2.imwrite(save_path, roi)
                    saved_count += 1
                    print(f"✅ [已蒐集] {save_path} | Total: {saved_count}")

                    # 更新 UI 顯示
                    last_result_text = f"Saved: {saved_count}"
                    last_result_color = (0, 255, 0) # 綠色
                    last_result_time = time.time()
                    
                    state = "CAPTURED"
                    trigger_cooldown = 45 # 拍完冷卻一下
                    stable_counter = 0
            else:
                stable_counter = 0
                if presence_area <= MIN_AREA_THRESHOLD:
                    state = "WAITING"
                    color = (100, 100, 100) # 灰色
                else:
                    state = "MOVING"
                    color = (255, 0, 0) # 紅色

        prev_gray = gray_blur.copy()

        # 6. UI 文字重置邏輯 (5秒後變回 Ready)
        if last_result_time != 0 and (time.time() - last_result_time > RESULT_RESET_SECONDS):
            last_result_text = "Ready..."
            last_result_color = (255, 255, 255)
            last_result_time = 0

        # 7. 繪圖 (保持與 main.py 完全一致的視覺參數)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 20)
        cv2.putText(frame, state, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)
        cv2.putText(frame, last_result_text, (x, y+h+100), cv2.FONT_HERSHEY_SIMPLEX, 3, last_result_color, 10)
        
        # 顯示畫面
        display_frame = cv2.resize(frame, (1024, 768))
        cv2.imshow('Data Collection Tool', display_frame)
        
        # 按鍵控制
        key = cv2.waitKey(30 if not USE_SENTECH_CAMERA else 1)
        if key == 27: break # ESC 離開
        elif key == ord('r'):
            # 手動重設背景
            avg_float = np.float32(gray_blur) 
            bg_path = f"{BG_SAVE_DIR}/background.png"
            cv2.imwrite(bg_path, roi)
            print(f">>> [指令] 背景已更新並儲存至: {bg_path} <<<")

finally:
    if USE_SENTECH_CAMERA and st_device:
        if st_device: st_device.acquisition_stop()
        if st_datastream: st_datastream.stop_acquisition()
    elif cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[*] 資料蒐集結束")