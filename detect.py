import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe Hands 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_drawing = mp.solutions.drawing_utils

# 影片路徑
video_path = 'video.mp4'
output_path = 'output_video.mp4'
# 讀取影片
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功打開
if not cap.isOpened():
    print('無法打開影片')
    exit()

# 取得影片的幀寬度、高度及每秒幀數
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定義 VideoWriter 物件，參數分別是輸出檔案名稱、編碼器、每秒幀數和幀大小
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 設定矩形框的左下角和右上角座標
Fixed_bottom_left = (100, 200)
Fixed_top_right = (300, 100)

Put_screws_bottom_left = (400, 200)
Put_screws_top_right = (600, 100)

Lock_screws_bottom_left = (100, 400)
Lock_screws_top_right = (300, 300)

Throw_bottom_left = (400, 400)
Throw_top_right = (600, 300)

Douyin_bottom_left = (700, 600)
Douyin_top_right = (900, 500)

text_position = (10, 30)
action_position = (10, 60)

state = 0
un_lo_state = 0
action_state = 0

# 讀取影片的幀並顯示
while True:
    ret, frame = cap.read()
    
    if not ret:
        print('影片讀取完畢')
        break

    # 轉換為HSV色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 固定 (Fixed) 檢測區域
    lower_color = np.array([90, 90, 90])
    upper_color = np.array([250, 250, 250])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=21)
    cv2.rectangle(mask, Fixed_bottom_left, Fixed_top_right, (0, 255, 0), 2)
    roi = mask[Fixed_top_right[1]:Fixed_bottom_left[1], Fixed_bottom_left[0]:Fixed_top_right[0]]
    mean_gray_value = np.mean(roi)
    if mean_gray_value > 5:
        cv2.putText(frame, 'Fixed', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        un_lo_state = 0
    else:
        cv2.putText(frame, 'Not Fixed', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        un_lo_state = 1

    # 手勢檢測
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for landmark in hand_landmarks.landmark:
                landmark_x = int(landmark.x * frame.shape[1])
                landmark_y = int(landmark.y * frame.shape[0])

                # 放螺絲 (Put screws) 檢測
                if (Put_screws_bottom_left[0] < landmark_x < Put_screws_top_right[0] and
                    Put_screws_top_right[1] < landmark_y < Put_screws_bottom_left[1]):
                    cv2.putText(frame, 'Put screws', action_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    action_state = 1
                    break
                
                # 鎖螺絲 (Lock Screws) 檢測
                if (Lock_screws_bottom_left[0] < landmark_x < Lock_screws_top_right[0] and
                    Lock_screws_top_right[1] < landmark_y < Lock_screws_bottom_left[1]):
                    cv2.putText(frame, 'Lock Screws', action_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    action_state = 2
                    break
                
                # 丟到產線 (Throw) 檢測
                if (Throw_bottom_left[0] < landmark_x < Throw_top_right[0] and
                    Throw_top_right[1] < landmark_y < Throw_bottom_left[1]):
                    cv2.putText(frame, 'Throw', action_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    action_state = 3
                    break
                
                # 滑抖音 (Scroll douyin) 檢測
                if (Douyin_bottom_left[0] < landmark_x < Douyin_top_right[0] and
                    Douyin_top_right[1] < landmark_y < Douyin_bottom_left[1]):
                    cv2.putText(frame, 'Scroll douyin', action_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    action_state = 4
                    break

    # 顯示處理後的畫面
    cv2.imshow('detect video', frame)
    if cv2.waitKey(25) != -1:
        break

# 釋放影片對象並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
