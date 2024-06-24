import datetime
from tkinter import filedialog
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
import mediapipe as mp
import numpy as np
import time

# 初始化 MediaPipe Hands 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

# =============================設置偵測範圍及初始狀態=============================

text_position = (10, 30)
text_position2 = (10, 60)
text_position3 = (10, 80)
text_position4 = (10, 100)
text_position5 = (10, 120)

# 初始化動作計時變數
start_time = None
current_action = None

# 初始化動作計時變數
last_time = 0
action_durations = {"Put_screws": 0, "Throw": 0, "lock_screw": 0, "Tiktok": 0}
steps = ["Put_screws", "lock_screw", "Throw"]
current_step = 0

# 設定矩形框的右下角和左上角座標
Put_screws_top_left = (445, 380)
Put_screws_bottom_right = (780, 530)
Lock_screws_top_left = (500, 200)
Lock_screws_bottom_right = (870, 360)
Throw_top_left = (970, 80)
Throw_bottom_right = (1300, 330)
Tiktok_top_left = (620, 600)
Tiktok_bottom_right = (1080, 880)

# 設定文字位置
text_position = (10, 30)
text_position2 = (10, 60)
text_position3 = (10, 80)
text_position4 = (10, 100)

# 初始化上一幀的時間
last_frame_time = time.time()
put_state = 0
throw_state = 0
lock_state = 0

state = 0
un_lo_state = 0
screw_state = 1

# ==================檢測流水線物品初始化變量=================== 
count =0
no_line_count = 0
max_no_line_count = 10
last_best_line = None
object_num = 0
objects = {}  # 物體的字典，鍵是物體ID，值是物體的位置信息
next_object_id = 0
obj_count = 0
last_state = False
reset_counts = False
last_times = [0,0]

video_path = 'assembly line.mp4'
cap = cv2.VideoCapture(video_path)

# 獲取第一幀並轉灰階
ret, old_frame = cap.read()
if not ret:
    print("Failed to read video")
    exit(1)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 設定直線檢測ROI區域為右上方
height, width = old_gray.shape
roi_old_gray = old_gray[0:height, width//2:width]

def extend_line(x1, y1, x2, y2, img_shape):
    height, width = img_shape[:2]
    if x1 == x2:
        return x1, 0, x2, height
    elif y1 == y2:
        return 0, y1, width, y2
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    x_ext1 = 0
    y_ext1 = int(intercept)
    x_ext2 = width
    y_ext2 = int(slope * width + intercept)

    if y_ext1 < 0:
        y_ext1 = 0
        x_ext1 = int(-intercept / slope)
    elif y_ext1 > height:
        y_ext1 = height
        x_ext1 = int((height - intercept) / slope)

    if y_ext2 < 0:
        y_ext2 = 0
        x_ext2 = int(-intercept / slope)
    elif y_ext2 > height:
        y_ext2 = height
        x_ext2 = int((height - intercept) / slope)

    return x_ext1, y_ext1, x_ext2, y_ext2

# =============================================================================

# 設定UI基礎介面
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.geometry("1400x600")
root.title("SOP Detect")

# =============================設定UI視頻框架==================================
video_frame = ctk.CTkFrame(master=root, width=500, height=400)
video_frame.grid(row=0, column=0, pady=0, padx=20)
image_label = ctk.CTkLabel(video_frame, text='', width=video_frame.cget("width"),
                           height=video_frame.cget("height") + 10)
image_label.grid(column=0, padx=50)

frame_list = []
play = 1
loadmp4 = False
current_value = 0.0
duration_seconds = 0
label_time_use_phone = 0
label_time_throw = 0

video_control_frame = ctk.CTkFrame(master=root, width=600, height=400)
video_control_frame.grid(row=1, column=0, pady=0, padx=20)
# ==========================================================================
def play_pause():
    global play
    play = 1 - play

def update_slider():
    global duration_seconds
    current_value = progress_value.get()
    if current_value < 1020:
        progress_value.set(current_value + 1)
        duration_seconds += 1

# 視頻控制元素
play_pause_btn = ctk.CTkButton(master=video_control_frame, text="Play/Pause", command=play_pause, width=20)
play_pause_btn.grid(row=0, column=0, pady=5, padx=20, sticky='w')

start_time_label = ctk.CTkLabel(video_control_frame, text=str(datetime.timedelta(seconds=0)))
start_time_label.grid(row=0, column=1, padx=20, sticky='nsew')

progress_value = tk.IntVar(video_control_frame)
progress_slider = ctk.CTkSlider(video_control_frame, variable=progress_value, from_=0, to=1020, orientation="horizontal")
progress_slider.grid(row=0, column=2)

end_time = ctk.CTkLabel(video_control_frame, text=str(datetime.timedelta(seconds=42)))
end_time.grid(row=0, column=3, padx=20)

# ===============================建立tabview分業==================================
tabview = ctk.CTkTabview(root, width=250, height=500)
tabview.grid(row=0, column=2, padx=(20, 10), pady=(10, 0), sticky="nsew")
tabview.add("計算與呈現")
tabview.add("動作設定")
tabview.add("步驟")
tabview.add("結果")
# ===============================================================================
# =================================分類動作==========================================
status_list = ['put screw', 'lock screw', 'throw', 'use phone']
status_setting_dictionary = {f'{status_list[i]}': {'bias_second': 0, 'reset_step': False} for i in range(len(status_list))}
status_order = [0 for i in range(len(status_list))]

video_path = None
cap = None
label_time = { 'put screw': 0, 'lock screw': 0, 'throw': 0, 'use phone': 0}
last_time_ = 0
counts = 0
isPhone = False
isThrow = False
def load_video_button_callback():
    global video_path, cap, loadmp4, frame_list, current_value, play, duration_seconds
    play = 0  # 暫停視頻
    loadmp4 = True
    file_path = filedialog.askopenfilename()
    if file_path:
        video_path = file_path
        cap = cv2.VideoCapture(video_path)
        frame_list = []
        current_value = 0
        duration_seconds = 0
        progress_value.set(0)
        load_video_button.configure(text="影片已上傳")
        play = 1


bbox = False  # 是否顯示bounding box

def bbox_checkbox_callback():
    global bbox
    if bbox_checkbox.get() == 'on':
        bbox = True
    else:
        bbox = False

# 計算與呈現元素
load_video_button = ctk.CTkButton(tabview.tab("計算與呈現"), text="上傳影片", width=70, font=("Roboto", 14), command=load_video_button_callback)
load_video_button.grid(row=0, column=0, padx=0, pady=10)

data_show_label = ctk.CTkLabel(tabview.tab("計算與呈現"), text="資料呈現方式", width=230, font=("Roboto", 14))
data_show_label.grid(row=2, column=0, padx=10, pady=0)

data_show_menu = ctk.CTkOptionMenu(tabview.tab("計算與呈現"), values=["顯示在待測物上"], width=230)
data_show_menu.grid(row=3, column=0, padx=20, pady=(0, 10))

apply_mode_label = ctk.CTkLabel(tabview.tab("計算與呈現"), text="應用模式", width=230, font=("Roboto", 14))
apply_mode_label.grid(row=4, column=0, padx=0, pady=0)

mode_select = ctk.CTkSegmentedButton(tabview.tab("計算與呈現"), values=["獨立計算", "獨立SOP", "共用SOP", "警報"], width=230)
mode_select.grid(row=5, column=0, padx=0, pady=0)

bbox_checkbox = ctk.CTkCheckBox(tabview.tab("計算與呈現"), text="顯示偵測框線", onvalue="on", offvalue="off", command=bbox_checkbox_callback)
bbox_checkbox.grid(row=6, column=0, padx=0, pady=20)

# 動作設定元素
plus_button_count = 0
label_text_button_list = []

def select_label_menu_callback(choice):
    bias_second_entry.configure(placeholder_text=0)

def bias_second_enter_callback():
    if bias_second_entry.get() != '':
        status_setting_dictionary[f'{select_label_menu.get()}']['bias_second'] = float(bias_second_entry.get())
    print(status_setting_dictionary)

def checkbox_callback():
    if checkbox.get() == 'on':
        status_setting_dictionary[f'{select_label_menu.get()}']['reset_step'] = True
    else:
        status_setting_dictionary[f'{select_label_menu.get()}']['reset_step'] = False
    print(status_setting_dictionary)

select_label = ctk.CTkLabel(tabview.tab("動作設定"), text="選擇狀態", width=60, font=("Roboto", 14))
select_label.grid(row=0, column=0, padx=0, pady=0)

select_label_menu = ctk.CTkOptionMenu(tabview.tab("動作設定"), values=status_list, width=230, command=select_label_menu_callback)
select_label_menu.grid(row=1, column=0, padx=20, pady=(0, 10))

bias_second_label = ctk.CTkLabel(tabview.tab("動作設定"), text="誤差秒數", width=60, font=("Roboto", 14))
bias_second_label.grid(row=2, column=0, padx=0, pady=0)

bias_second_entry = ctk.CTkEntry(tabview.tab("動作設定"), width=40, height=20, border_width=2, corner_radius=0)
bias_second_entry.grid(row=3, column=0, padx=0, pady=5)
bias_second_entry.configure(placeholder_text=0)

bias_second_enter = ctk.CTkButton(tabview.tab("動作設定"), text="Enter", width=20, height=25, border_width=2, corner_radius=10, command=bias_second_enter_callback)
bias_second_enter.grid(row=4, column=0, padx=20, pady=5)

checkbox = ctk.CTkCheckBox(tabview.tab("動作設定"), text="設為重製步驟", onvalue="on", offvalue="off", command=checkbox_callback)
checkbox.grid(row=5, column=0, padx=0, pady=5)

# 步驟元素
step_count = 0

def add_step_button_callback():
    global step_count
    if step_count < len(status_list):
        selected_status = select_status.get()
        status_order[step_count] = (selected_status)
        step_count += 1
        print(status_order)
        steps_label = ctk.CTkLabel(master=step_display_frame, text=f"{step_count}. {selected_status}", fg_color="gray", text_color="white", corner_radius=5, width=200)
        steps_label.grid(pady=5, padx=0)

def finish_button_callback():
    global cap, video_path, duration_seconds
    if finish_button.cget('text') == "設定完成":
        finish_button.configure(text='已完成設定')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    progress_value.set(0)
    frame_list.clear()
    duration_seconds = 0

status_label = ctk.CTkLabel(tabview.tab("步驟"), text="選擇狀態", width=230, font=("Roboto", 14))
status_label.grid(row=2, column=0, padx=20, pady=0)
select_status = ctk.CTkOptionMenu(tabview.tab("步驟"), values=status_list, width=230)
select_status.grid(row=3, column=0, padx=20, pady=(0, 10))
select_status.set(f"{status_list[0]}")

add_step_button = ctk.CTkButton(tabview.tab("步驟"), text="+新增步驟", command=add_step_button_callback, width=230)
add_step_button.grid(row=4, column=0, padx=20, pady=5)

step_display_frame = ctk.CTkScrollableFrame(tabview.tab("步驟"), width=230)
step_display_frame.grid(row=5, column=0, padx=20, pady=5)
step_display_frame.grid_columnconfigure(0, weight=1)

finish_button = ctk.CTkButton(tabview.tab("步驟"), text="設定完成", width=70, command=finish_button_callback)
finish_button.grid(row=6, column=0)

# ==========================================結果元素==========================================

select_label = ctk.CTkLabel(tabview.tab("結果"), text="持續時間", width=60, font=("Roboto", 14))
select_label.grid(row=1, column=0, padx=0, pady=10)

add_step_button_all = ctk.CTkButton(tabview.tab("結果"), text="0", width=150, hover=False)
add_step_button_all.grid(row=1, column=1, padx=30, pady=0)

select_label0 = ctk.CTkLabel(tabview.tab("結果"), text=f"{status_order[0]}", width=60, font=("Roboto", 14))
select_label0.configure(text=f'{status_order[0]}')
select_label0.grid(row=2, column=0, padx=0, pady=10)

add_step_button_label1 = ctk.CTkButton(tabview.tab("結果"), text="0", width=150, hover=False)
add_step_button_label1.grid(row=2, column=1, padx=30, pady=0)

select_label1 = ctk.CTkLabel(tabview.tab("結果"), text=f"{status_order[1]}", width=60, font=("Roboto", 14))
select_label1.configure(text=f'{status_order[1]}')
select_label1.grid(row=3, column=0, padx=0, pady=10)

add_step_button_label2 = ctk.CTkButton(tabview.tab("結果"), text="0", width=150, hover=False)
add_step_button_label2.grid(row=3, column=1, padx=30, pady=0)

select_label2 = ctk.CTkLabel(tabview.tab("結果"), text=f"{status_order[2]}", width=60, font=("Roboto", 14))
select_label2.configure(text=f'{status_order[2]}')
select_label2.grid(row=4, column=0, padx=0, pady=10)

add_step_button_label3 = ctk.CTkButton(tabview.tab("結果"), text="0", width=150, hover=False)
add_step_button_label3.grid(row=4, column=1, padx=30, pady=0)

select_label3 = ctk.CTkLabel(tabview.tab("結果"), text=f"{status_order[3]}", width=60, font=("Roboto", 14))
select_label3.configure(text=f'{status_order[3]}')
select_label3.grid(row=5, column=0, padx=0, pady=10)

add_step_button_label4 = ctk.CTkButton(tabview.tab("結果"), text="0", width=150, hover=False)
add_step_button_label4.grid(row=5, column=1, padx=30, pady=0)

latest_time_label = ctk.CTkLabel(tabview.tab("結果"), text="單輪執行時間", width=60, font=("Roboto", 14))
latest_time_label.grid(row=6, column=0, padx=0, pady=10)

latest_time_button = ctk.CTkButton(tabview.tab("結果"), text="0", width=150, hover=False)
latest_time_button.grid(row=6, column=1, padx=30, pady=0)

work_counts_label = ctk.CTkLabel(tabview.tab("結果"), text="已完成次數", width=60, font=("Roboto", 14))
work_counts_label.grid(row=7, column=0, padx=0, pady=10)

work_counts_button = ctk.CTkButton(tabview.tab("結果"), text=str(counts), width=60, hover=False)
work_counts_button.grid(row=7, column=1, padx=30, pady=0)

object_counts_label = ctk.CTkLabel(tabview.tab("結果"), text="流水線經過完成品", width=60, font=("Roboto", 14))
object_counts_label.grid(row=8, column=0, padx=0, pady=10)

object_counts_button = ctk.CTkButton(tabview.tab("結果"), text=str(object_num), width=60, hover=False)
object_counts_button.grid(row=8, column=1, padx=30, pady=0)


#===================================UI結束================================================



# 更新幀
def update_results():
    if(last_times[-1]!=last_time_):
        last_times.append(last_time_)
    #print(last_time_,last_times[-1])
    # 更新 "結果" 頁面的資料
    add_step_button_all.configure(text=str(np.around(last_time/25,1)))
    select_label0.configure(text=f"{status_order[0]}")
    add_step_button_label1.configure(text=str(np.around(label_time[status_order[0]]/25,1)))
    select_label1.configure(text=f"{status_order[1]}")
    add_step_button_label2.configure(text=str(np.around(label_time[status_order[1]]/25,1)))
    select_label2.configure(text=f"{status_order[2]}")
    add_step_button_label3.configure(text=str(np.around(label_time_throw/25,1)))
    select_label3.configure(text=f"{status_order[3]}")
    add_step_button_label4.configure(text=str(np.around(label_time_use_phone/25,1)))
    latest_time_button.configure(text=str(np.around(last_time_-last_times[-2],1)))
    work_counts_button.configure(text=str(counts))
    object_counts_button.configure(text=str(object_num))
    tabview.tab("結果").update()
    

def update_frame():
    global current_value, cap, screw_state, no_line_count, current_action, start_time, last_best_line, objects, object_num, next_object_id,last_frame_time,current_step
    global last_time,label_time_use_phone,isPhone, counts, count, obj_count,last_state,reset_counts , last_time_,last_time,label_time_throw,isThrow,last_times,sec
    try:
        if finish_button.cget('text') == '設定完成':
            if cap is not None and play == 1:
                success, frame = cap.read()
                if success:
                    current_action = None
                    current_value = progress_value.get()
                    frame_list.append(frame)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (960, 540))
                    img = Image.fromarray(img)
                    img_tk = ImageTk.PhotoImage(image=img)
                    image_label.configure(image=img_tk)
                    image_label.image = img_tk
                    update_slider()
                    root.update()
                    video_control_frame.update()
                    video_frame.update()
                else:
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
        elif finish_button.cget('text') == '已完成設定':
            if play == 1:
                success, frame = cap.read()
                init_frame = frame.copy()
                if success:
                    if last_time == 0:
                        tabview.set("結果")
                    last_time += 1
                    # 轉為灰階
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # 設置ROI區域為右上方
                    roi_frame_gray = frame_gray[0:height, width//2:width]

                    # 進行高斯模糊
                    roi_frame_gray = cv2.GaussianBlur(roi_frame_gray, (13, 13), 0)

                    # 使用Canny邊緣檢測
                    edges = cv2.Canny(roi_frame_gray, 50, 150)

                    # 膨脹和腐蝕操作
                    edges = cv2.dilate(edges, None, iterations=9)
                    edges = cv2.erode(edges, None, iterations=9)

                    # 使用霍夫轉換来檢測邊緣
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=300, maxLineGap=5)

                    # 選擇最顯著的一條邊緣線，且斜率大於0
                    if lines is not None:
                        max_len = 0
                        best_line = None
                        for line in lines:
                            for x1, y1, x2, y2 in line:
                                if x2 - x1 != 0:
                                    slope = (y2 - y1) / (x2 - x1)
                                    if slope > 0:  # 斜率大於0
                                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                        if length > max_len:
                                            max_len = length
                                            best_line = (x1, y1, x2, y2)
                        if best_line is not None:
                            x1, y1, x2, y2 = best_line
                            x1 += width // 2  # 調整點的位置以匹配原始圖像座標
                            x2 += width // 2  # 調整點的位置以匹配原始圖像座標
                            # 延伸線到畫面邊缘
                            x1_ext, y1_ext, x2_ext, y2_ext = extend_line(x1, y1, x2, y2, frame.shape)
                            last_best_line = (x1_ext, y1_ext, x2_ext, y2_ext)
                            no_line_count = 0
                        else:
                            no_line_count += 1
                    else:
                        no_line_count += 1

                    

                    if last_best_line is not None and no_line_count < max_no_line_count:
                        x1_ext, y1_ext, x2_ext, y2_ext = last_best_line
                        cv2.line(frame, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 0, 255), 4)

                        # 在紅線中間畫一條黄色的垂直線，僅延伸到红線以上
                        mid_x = (x1_ext + x2_ext) // 2
                        mid_y = (y1_ext + y2_ext) // 2

                        # 垂直於紅線的方向
                        if x1_ext != x2_ext:
                            perp_slope = -1 / ((y2_ext - y1_ext) / (x2_ext - x1_ext))
                            intercept = mid_y - perp_slope * mid_x

                            y_end = 0
                            x_end = int((y_end - intercept) / perp_slope)
                            cv2.line(frame, (mid_x, mid_y), (x_end, y_end), (0, 255, 255), 2)
                        else:
                            cv2.line(frame, (mid_x, mid_y), (mid_x, 0), (0, 255, 255), 2)

                        # 顏色過濾以提取黑色盒子
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_black = np.array([0, 0, 0])
                        upper_black = np.array([180, 255, 50])
                        mask_black = cv2.inRange(hsv, lower_black, upper_black)
                        

                        # 提取黑色區域
                        res_black = cv2.bitwise_and(frame, frame, mask=mask_black)

                        # 檢測黑色盒子的輪廓
                        contours, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        new_objects = {}

                        for contour in contours:
                            if cv2.contourArea(contour) > 500:  # 閥值
                                x, y, w, h = cv2.boundingRect(contour)
                                if x > mid_x and y < mid_y:
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    
                                    # 檢測當前物體是否为新物體
                                    found_existing = False
                                    for obj_id, (ox, oy, ow, oh) in objects.items():
                                        if np.sqrt((ox - x)**2 + (oy - y)**2) < 1000 :
                                            # 更新物體位置
                                            new_objects[obj_id] = (x, y, x + w, y + h)
                                            found_existing = True
                                            break
                                    if not found_existing:
                                        if 2 <= object_num < 3:
                                            count += 1
                                            if count % 3 == 0:  # 每兩次增加一次num
                                                object_num += 1
                                        else:
                                            object_num += 1
                                        new_objects[next_object_id] = (x, y, x + w, y + h)
                                        next_object_id += 1
                        objects = new_objects
# ===============================流水線物品檢測結束===============================================
# ==================================SOP檢測開始===============================================

                    #cv2.rectangle(frame, Put_screws_top_left, Put_screws_bottom_right, (0, 0, 255), 2)  # Throw
                    #cv2.rectangle(frame, Lock_screws_top_left, Lock_screws_bottom_right, (0, 0, 255), 2)  # Throw
                    #cv2.rectangle(frame, Throw_top_left, Throw_bottom_right, (0, 0, 255), 2)  # Throw
                    #cv2.rectangle(frame, Tiktok_top_left, Tiktok_bottom_right, (0, 0, 255), 2)  # Throw
                    # 計算當前幀的持續時間

                    current_frame_time = time.time()

                    last_frame_time = current_frame_time




                    

                    # 如果有當前動作，將幀持續時間加到該動作的總時間中
                    if current_action:
                        label_time[current_action] += 1
                    current_action = None
                    
                    
                    if last_state == False:
                        if reset_counts == True:
                            counts += 1
                            last_state = True
                            last_time_ = (last_time - float(latest_time_button.cget("text")))/25
                                


                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb_frame)

                    if result.multi_hand_landmarks and result.multi_handedness:
                        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                            hand_label = result.multi_handedness[idx].classification[0].label
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                            for landmark in hand_landmarks.landmark:
                                landmark_x = int(landmark.x * frame.shape[1])
                                landmark_y = int(landmark.y * frame.shape[0])

                                if (Tiktok_top_left[0] < landmark_x < Tiktok_bottom_right[0] and
                                    Tiktok_top_left[1] < landmark_y < Tiktok_bottom_right[1]):
                                    # cv2.putText(frame, "Tiktok", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                    isPhone = True
                                    reset_counts = False
                                    variable_name = "add_step_button_label{}".format(4)
                                    label_widget = globals()[variable_name]
                                    label_widget.configure(fg_color='red')
                                    for i in (1, 2, 3):
                                        variable_name = "add_step_button_label{}".format(i)
                                        label_widget = globals()[variable_name]
                                        label_widget.configure(fg_color=['#3B8ED0', '#1F6AA5'])
                                    start_time = time.time()
                                        
                                if Throw_top_left[0] < landmark_x < Throw_bottom_right[0] and Throw_top_left[1] < landmark_y < Throw_bottom_right[1]:
                                    # cv2.putText(frame, "Throw", text_position4, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                    isThrow = True
                                    if reset_counts != True:
                                        reset_counts = True
                                        last_state = False
                                    variable_name = "add_step_button_label{}".format(3)
                                    label_widget = globals()[variable_name]
                                    label_widget.configure(fg_color='green')
                                    for i in (1, 2, 4):
                                        variable_name = "add_step_button_label{}".format(i)
                                        label_widget = globals()[variable_name]
                                        label_widget.configure(fg_color=['#3B8ED0', '#1F6AA5'])
                                    start_time = time.time()

                                if current_step < len(steps):
                                    action = steps[current_step]

                                    if (action == "Put_screws" and Put_screws_top_left[0] < landmark_x < Put_screws_bottom_right[0] and
                                         Put_screws_top_left[1] < landmark_y < Put_screws_bottom_right[1] and current_action != "lock screw" and hand_label == "Left"):
                                        # cv2.putText(frame, "Put Screws", text_position2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                        if current_action != "put screw":
                                            reset_counts = False
                                            current_action = "put screw"
                                            variable_name = "add_step_button_label{}".format(1)
                                            label_widget = globals()[variable_name]
                                            label_widget.configure(fg_color='green')
                                            for i in (2, 3, 4):
                                                variable_name = "add_step_button_label{}".format(i)
                                                label_widget = globals()[variable_name]
                                                label_widget.configure(fg_color=['#3B8ED0', '#1F6AA5'])
                                            start_time = time.time()
                                    elif action == "lock_screw" and Lock_screws_top_left[0] < landmark_x < Lock_screws_bottom_right[0] and Lock_screws_top_left[1] < landmark_y < Lock_screws_bottom_right[1] and hand_label == "Right":
                                        # cv2.putText(frame, "Lock Screw", text_position3, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                        if current_action != "lock screw":
                                            reset_counts = False
                                            current_action = "lock screw"
                                            variable_name = "add_step_button_label{}".format(2)
                                            label_widget = globals()[variable_name]
                                            label_widget.configure(fg_color='green')
                                            for i in (1, 3, 4):
                                                variable_name = "add_step_button_label{}".format(i)
                                                label_widget = globals()[variable_name]
                                                label_widget.configure(fg_color=['#3B8ED0', '#1F6AA5'])
                                            start_time = time.time()
                                    if action == "Throw" and Throw_top_left[0] < landmark_x < Throw_bottom_right[0] and Throw_top_left[1] < landmark_y < Throw_bottom_right[1] and hand_label == "Right":
                                        # cv2.putText(frame, "Throw", text_position4, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                        if current_action != "throw":
                                            if reset_counts != True:
                                                reset_counts = True
                                            current_action = "throw"
                                            variable_name = "add_step_button_label{}".format(3)
                                            label_widget = globals()[variable_name]
                                            label_widget.configure(fg_color='green')
                                            for i in (1, 2, 4):
                                                variable_name = "add_step_button_label{}".format(i)
                                                label_widget = globals()[variable_name]
                                                label_widget.configure(fg_color=['#3B8ED0', '#1F6AA5'])
                                            start_time = time.time()
                                        


                                # 若當前動作已達成，進入下一步驟
                                if start_time and time.time() - start_time > 5:
                                    current_step += 1
                                    current_step %= 3
                                    start_time = None
                                    current_action = None
                                    break

                    if isPhone:
                        label_time_use_phone += 1
                    if isThrow:
                        label_time_throw += 1
                    isPhone = False
                    isThrow = False
                    if bbox == True :
                        Dframe = frame.copy()
                    elif bbox == False:
                        Dframe = init_frame
                    
                    img = cv2.cvtColor(Dframe, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (960, 540))
                    img = Image.fromarray(img)
                    img_tk = ImageTk.PhotoImage(image=img)
                    image_label.configure(image=img_tk)
                    image_label.image = img_tk
                    update_slider()
                    root.update()
                    video_control_frame.update()
                    video_frame.update()
                    update_results()

    except Exception as e:
        print(f"An error occurred: {e}")
    root.after(30, update_frame)

root.after(30, update_frame)
root.mainloop()