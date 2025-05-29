# import cv2
# import numpy as np
# import time
# from pyModbusTCP.client import ModbusClient

# # 색상별 HSV 범위 정의 (필요에 따라 조정)
# COLOR_RANGES = {
#     'red':    ([0, 100, 20], [10, 255, 255]),
#     'orange': ([10, 100, 20], [25, 255, 255]),
#     'yellow': ([25, 100, 100], [35, 255, 255]),
#     'green':  ([40, 50, 50], [80, 255, 255]),
#     'blue':   ([90, 100, 100], [140, 255, 255]),
#     'purple': ([140, 100, 100], [160, 255, 255])
# }
# #UR모드버스 주소
# ur_modbus = ModbusClient(host="192.168.213.78", port=502, auto_open=True)

# def create_mask(hsv_frame, lower, upper):
#     mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     return mask

# def detect_shape(cnt):
#     epsilon = 0.04 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     vertices = len(approx)
#     shape = "unknown"
#     if vertices == 3:
#         shape = "triangle"
#     elif vertices == 4:
#         x, y, w, h = cv2.boundingRect(approx)
#         aspect_ratio = float(w) / h
#         if 0.95 < aspect_ratio < 1.05:
#             shape = "square"
#         else:
#             shape = "rectangle"
#         # 마름모 판별 (사각형이지만 각도가 기울어진 경우)
#         pts = approx.reshape(4, 2)
#         vec1 = pts[1] - pts[0]
#         vec2 = pts[2] - pts[1]
#         angle = np.abs(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
#         if 0.7 < angle < 1.0:  # 약 40~60도
#             shape = "diamond"
#     elif vertices == 5:
#         shape = "pentagon"
#     elif vertices > 8:
#         shape = "circle"
#     return shape

# def send_ur_shape():
#     ur_modbus.write_single_coils(130, vertices)
#     time.sleep(0.1)

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FPS, 30)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     for color, (lower, upper) in COLOR_RANGES.items():
#         mask = create_mask(hsv, lower, upper)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < 1000:
#                 continue
#             shape = detect_shape(cnt)
#             if shape == "unknown":
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.putText(frame, f"{color} {shape}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.drawContours(frame, [cnt], 0, (0,255,0), 2)

#     cv2.imshow('Multi-Color Shape Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

"""
Multi-color Shape Detection & Modbus Communication System
Version 2.1 (2025-05-10)
Features: Enhanced error handling, improved shape detection, modular architecture
"""

"""
Multi-color Shape Detection & Modbus Communication System
Version 2.2 (2025-05-10)
Features: Split color/shape registers, enhanced stability
"""

import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusIOException
import logging
import time
import sys

# ******************* 시스템 설정 *******************
#region CONFIGURATION

# 카메라 설정
CAMERA_INDEX = 1                  # 시스템 환경에 맞게 변경
FRAME_WIDTH = 640                 # 처리 프레임 폭
FRAME_HEIGHT = 480                # 처리 프레임 높이
TARGET_FPS = 30                   # 목표 프레임 레이트

# Modbus 설정
MODBUS_IP = "192.168.213.78"      # UR 컨트롤러 IP
MODBUS_PORT = 502                 # 기본 Modbus 포트
COLOR_REGISTER = 130              # 색상 데이터 레지스터 주소
SHAPE_REGISTER = 131              # 도형 데이터 레지스터 주소
RECONNECT_INTERVAL = 5            # 재접속 시도 간격(초)

# 색상 HSV 범위 (H: 0-180, S: 0-255, V: 0-255)
COLOR_RANGES = {
    'red': ([0, 150, 100], [5, 255, 255], [175, 150, 100], [180, 255, 255]),
    'orange': ([6, 150, 50], [20, 255, 255]),
    'yellow': ([21, 150, 150], [35, 255, 255]),
    'green':  ([40, 50, 50], [80, 255, 255]),
    'blue':   ([90, 150, 50], [130, 255, 255]),
    'purple': ([131, 150, 50], [160, 255, 255])
}

# 코드 매핑 테이블
COLOR_CODE = {'red':1, 'orange':2, 'yellow':3, 'green':4, 'blue':5, 'purple':6}
SHAPE_CODE = {'triangle':1, 'square':2, 'rectangle':3, 
             'diamond':4, 'pentagon':5, 'circle':6}

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('detection.log'), logging.StreamHandler()]
)

#endregion
# ***********************************************

class ModbusHandler:
    """Modbus TCP 통신을 관리하는 클래스"""
    def __init__(self):
        self.client = None
        self.last_connect_time = 0
        self.connect()

    def connect(self):
        """Modbus 서버에 연결 시도"""
        try:
            if time.time() - self.last_connect_time > RECONNECT_INTERVAL:
                self.client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
                if self.client.connect():
                    logging.info("Modbus 연결 성공")
                    return True
                raise ConnectionError("연결 실패")
        except Exception as e:
            logging.error(f"Modbus 연결 오류: {str(e)}")
            self.client = None
        return False

    def write_registers(self, color_value, shape_value):
        """두 레지스터에 동시 쓰기"""
        try:
            if not self.client or not self.client.connected:
                if not self.connect():
                    return False
                    
            # 색상 레지스터 쓰기
            color_result = self.client.write_register(COLOR_REGISTER, color_value)
            # 도형 레지스터 쓰기
            shape_result = self.client.write_register(SHAPE_REGISTER, shape_value)
            
            return not (color_result.isError() or shape_result.isError())
            
        except ModbusIOException as e:
            logging.error(f"쓰기 오류: {str(e)}")
            self.client.close()
            return False

    def close(self):
        """연결 종료"""
        if self.client and self.client.connected:
            self.client.close()
            logging.info("Modbus 연결 종료")

def create_mask(hsv_frame, color_range):
    """
    HSV 범위 기반 마스크 생성
    :param hsv_frame: HSV 변환된 프레임
    :param color_range: 해당 색상 범위 튜플
    :return: 결합된 마스크
    """
    masks = []
    for i in range(0, len(color_range), 2):
        lower = np.array(color_range[i])
        upper = np.array(color_range[i+1])
        masks.append(cv2.inRange(hsv_frame, lower, upper))
    
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
    return processed_mask

def detect_shape(contour):
    """
    컨투어 기반 도형 판별
    :param contour: 감지된 컨투어
    :return: 도형 이름 문자열
    """
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # 삼각형 판별
    if vertices == 3:
        return 'triangle'
    
    # 사각형 계열 판별
    if vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # 정사각형/직사각형 판별
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        else:
            # 마름모 판별
            pts = approx.reshape(4, 2)
            vectors = [pts[i] - pts[i-1] for i in range(4)]
            angles = []
            for i in range(4):
                v1 = vectors[i]
                v2 = vectors[(i+1)%4]
                dot = np.dot(v1, v2)
                cos_theta = dot / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-10)
                angles.append(np.arccos(np.clip(cos_theta, -1, 1)))
            
            if all(np.degrees(angle) > 80 and np.degrees(angle) < 100 for angle in angles):
                return 'diamond'
            return 'rectangle'
    
    # 오각형 판별
    if vertices == 5:
        return 'pentagon'
    
    # 원 판별 (허프 변환 대체 방법)
    area = cv2.contourArea(contour)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    if abs(1 - (area / circle_area)) < 0.2:
        return 'circle'
    
    return 'unknown'

def main():
    """메인 처리 루프"""
    # 카메라 초기화
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    if not cap.isOpened():
        logging.error("카메라 초기화 실패")
        sys.exit(1)
    
    # Modbus 핸들러 생성
    modbus = ModbusHandler()
    last_sent = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("프레임 읽기 실패")
                continue
            
            # 프레임 전처리
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            detected = False
            
            for color_name, color_range in COLOR_RANGES.items():
                mask = create_mask(hsv, color_range)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 1500:  # 최소 영역 필터링
                        continue
                    
                    shape = detect_shape(cnt)
                    if shape == 'unknown':
                        continue
                    
                    # 화면 표시
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{color_name} {shape}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    # Modbus 전송
                    if color_name in COLOR_CODE and shape in SHAPE_CODE:
                        color_val = COLOR_CODE[color_name]
                        shape_val = SHAPE_CODE[shape]
                        
                        if modbus.write_registers(color_val, shape_val):
                            if time.time() - last_sent > 1:  # 1초 간격 로깅
                                logging.info(f"전송 성공: [색상:{color_name}({color_val})] → 레지스터 {COLOR_REGISTER}, [도형:{shape}({shape_val})] → 레지스터 {SHAPE_REGISTER}")
                                last_sent = time.time()
                        detected = True
            
            # 화면 출력
            cv2.imshow('Object Detection', frame)
            
            # 종료 처리
            if cv2.waitKey(max(1, int(1000/TARGET_FPS))) & 0xFF == ord('q'):
                logging.info("사용자 종료 요청")
                break
            
    except KeyboardInterrupt:
        logging.info("시스템 종료")
    finally:
        cap.release()
        modbus.close()
        cv2.destroyAllWindows()
        logging.info("시스템 리소스 정리 완료")

if __name__ == "__main__":
    main()

# import cv2
# import mediapipe as mp
# from pymodbus.client import ModbusTcpClient
# import time
# import absl.logging

# # 로그 레벨 낮추기 (경고 메시지 최소화)
# absl.logging.set_verbosity(absl.logging.ERROR)

# # MODBUS 설정
# client = ModbusTcpClient("192.168.213.78")
# client.connect()

# # MediaPipe 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     model_complexity=0,  # Lite 모델로 메모리/속도 최적화
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils

# # 카메라 설정
# cap = cv2.VideoCapture(1)  # 0 또는 1로 맞게 조정
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# def count_fingers(hand_landmarks, hand_label):
#     tips = [4, 8, 12, 16, 20]
#     fingers = []

#     # 엄지
#     if hand_label == "Left":
#         if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     else:
#         if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)

#     # 나머지 4개 손가락
#     for tip in tips[1:]:
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
#             fingers.append(1)
#         else:
#             fingers.append(0)

#     return sum(fingers)

# last_send_time = 0
# send_interval = 1.0  # 1초에 한 번만 MODBUS 전송

# while True:
#     ret, img = cap.read()
#     if not ret:
#         print("카메라 프레임을 읽을 수 없습니다.")
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     height, width = img.shape[:2]

#     # MediaPipe에 이미지 크기 정보 전달 (경고 최소화)
#     # (실제로는 Hands API에서 직접적으로 image_size를 받지 않지만, 경고 무시 목적)
#     # hands._input_side_packets = {'image_size': (width, height)}  # 최신 mediapipe에서는 무시해도 됨

#     results = hands.process(img_rgb)

#     if results.multi_hand_landmarks and results.multi_handedness:
#         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#             # handedness.classification이 비어있는 경우 예외 방지
#             if not handedness.classification:
#                 continue
#             hand_label = handedness.classification[0].label  # "Left" or "Right"
#             mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             finger_count = count_fingers(hand_landmarks, hand_label)
#             y_pos = 50 if hand_label == "Left" else 100
#             cv2.putText(img, f'{hand_label} Fingers: {finger_count}',
#                         (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # MODBUS 명령 전송 (1초에 한 번만, 예외 처리)
#             now = time.time()
#             try:
#                 if client.connected and (now - last_send_time) > send_interval:
#                     if finger_count == 1:
#                         client.write_register(133, 0)
#                     elif finger_count == 2:
#                         client.write_register(133, 1)
#                     elif finger_count == 5:
#                         client.write_register(133, 2)
#                     last_send_time = now
#             except Exception as e:
#                 print(f"MODBUS 통신 오류: {e}")
#                 # 연결이 끊겼다면 재연결 시도
#                 client.close()
#                 time.sleep(0.5)
#                 client.connect()

#     cv2.imshow("Hand Detection", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# client.close()
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# from pymodbus.client import ModbusTcpClient
# import time
# import absl.logging
# from collections import Counter

# # 로그 레벨 낮추기 (경고 메시지 최소화)
# absl.logging.set_verbosity(absl.logging.ERROR)

# # MODBUS 설정
# client = ModbusTcpClient("192.168.213.78")
# client.connect()

# # MediaPipe 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     model_complexity=0,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils

# # 카메라 설정
# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # HSV 색상 범위
# COLOR_RANGES = {
#     'red': ([0, 150, 100], [5, 255, 255], [175, 150, 100], [180, 255, 255]),
#     'orange': ([6, 150, 50], [20, 255, 255]),
#     'yellow': ([21, 150, 150], [35, 255, 255]),
#     'green':  ([40, 50, 50], [80, 255, 255]),
#     'blue':   ([90, 150, 50], [130, 255, 255]),
#     'purple': ([131, 150, 50], [160, 255, 255])
# }
# COLOR_CODE = {'red': 1, 'orange': 2, 'yellow': 3, 'green': 4, 'blue': 5, 'purple': 6}

# # 도형 인식 함수
# def detect_shape(contour):
#     epsilon = 0.04 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     vertices = len(approx)
#     if vertices == 3:
#         return 1  # 삼각형
#     elif vertices == 4:
#         (x, y, w, h) = cv2.boundingRect(approx)
#         aspect_ratio = w / float(h)
#         if 0.9 <= aspect_ratio <= 1.1:
#             return 2  # 정사각형
#         else:
#             return 3  # 직사각형
#     elif vertices == 5:
#         return 4  # 오각형
#     else:
#         area = cv2.contourArea(contour)
#         (_, _), radius = cv2.minEnclosingCircle(contour)
#         circle_area = np.pi * (radius ** 2)
#         if abs(1 - (area / circle_area)) < 0.2:
#             return 5  # 원
#     return 0  # 알 수 없는 도형

# # 손가락 개수 세기 함수
# def count_fingers(hand_landmarks, hand_label):
#     tips = [4, 8, 12, 16, 20]
#     fingers = []
#     # 엄지 처리 (flip된 이미지 기준)
#     if hand_label == "Left":
#         if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     else:
#         if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     # 나머지 손가락
#     for tip in tips[1:]:
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     return sum(fingers)

# # 색상 마스크 생성 함수
# def create_mask(hsv_frame, color_range):
#     masks = []
#     for i in range(0, len(color_range), 2):
#         lower = np.array(color_range[i])
#         upper = np.array(color_range[i+1])
#         masks.append(cv2.inRange(hsv_frame, lower, upper))
#     combined_mask = masks[0]
#     for mask in masks[1:]:
#         combined_mask = cv2.bitwise_or(combined_mask, mask)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
#     processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
#     return processed_mask

# # 인식 제어 변수
# detecting = False
# detect_type = None  # 'shape_count', 'color', 'shape_kind'
# detect_start = 0
# detect_duration = 2  # 초

# while True:
#     ret, img = cap.read()
#     if not ret:
#         print("카메라 프레임을 읽을 수 없습니다.")
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)

#     finger_count = 0

#     if results.multi_hand_landmarks and results.multi_handedness:
#         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#             if not handedness.classification:
#                 continue
#             hand_label = handedness.classification[0].label
#             mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             finger_count = count_fingers(hand_landmarks, hand_label)
#             y_pos = 50 if hand_label == "Left" else 100
#             cv2.putText(img, f'{hand_label} Fingers: {finger_count}', (50, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     now = time.time()
#     try:
#         # 손가락 1개: 도형 개수 인식
#         if finger_count == 1 and not detecting:
#             detecting = True
#             detect_type = 'shape_count'
#             detect_start = now
#             print("손가락 1개: 도형 개수 인식 시작")
#         # 손가락 2개: 색상 인식
#         elif finger_count == 2 and not detecting:
#             detecting = True
#             detect_type = 'color'
#             detect_start = now
#             print("손가락 2개: 색상 인식 시작")
#         # 손가락 3개: 도형 모양 인식
#         elif finger_count == 3 and not detecting:
#             detecting = True
#             detect_type = 'shape_kind'
#             detect_start = now
#             print("손가락 3개: 도형 모양 인식 시작")

#         # 인식 중일 때
#         if detecting and (now - detect_start) < detect_duration:
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             detected_shapes = []
#             detected_colors = []
#             for color_name, color_range in COLOR_RANGES.items():
#                 mask = create_mask(hsv, color_range)
#                 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 for cnt in contours:
#                     area = cv2.contourArea(cnt)
#                     if area < 1500:
#                         continue
#                     shape_num = detect_shape(cnt)
#                     if shape_num == 0:
#                         continue
#                     detected_shapes.append(shape_num)
#                     detected_colors.append(COLOR_CODE[color_name])
#                     x, y, w, h = cv2.boundingRect(cnt)
#                     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(img, f'{color_name} {shape_num}', (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # 2초간 누적된 인식 결과를 2초가 지난 뒤 전송
#             if (now - detect_start) >= detect_duration - 0.1:
#                 if detect_type == 'shape_count':
#                     count_to_send = len(detected_shapes)
#                     client.write_register(135, count_to_send)
#                     print(f"도형 개수 {count_to_send}개 → 레지스터 135로 전송")
#                 elif detect_type == 'color':
#                     if detected_colors:
#                         color_to_send = Counter(detected_colors).most_common(1)[0][0]
#                         client.write_register(130, color_to_send)
#                         print(f"색상 코드 {color_to_send} → 레지스터 130으로 전송")
#                     else:
#                         client.write_register(130, 0)
#                         print("색상 없음: 0 전송")
#                 elif detect_type == 'shape_kind':
#                     if detected_shapes:
#                         shape_to_send = Counter(detected_shapes).most_common(1)[0][0]
#                         client.write_register(131, shape_to_send)
#                         print(f"도형 모양 코드 {shape_to_send} → 레지스터 131로 전송")
#                     else:
#                         client.write_register(131, 0)
#                         print("도형 없음: 0 전송")
#                 detecting = False
#                 detect_type = None

#         # 손가락이 1,2,3이 아니면 인식 모드 종료
#         if finger_count not in [1,2,3]:
#             detecting = False
#             detect_type = None

#     except Exception as e:
#         print(f"MODBUS 통신 오류: {e}")
#         client.close()
#         time.sleep(0.5)
#         client.connect()

#     cv2.imshow("Hand/Shape/Color Detection", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# client.close()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# from pymodbus.client import ModbusTcpClient
# import time
# import absl.logging
# from collections import Counter
# import threading

# absl.logging.set_verbosity(absl.logging.ERROR)

# # MODBUS 설정
# client = ModbusTcpClient("192.168.213.78")
# client.connect()

# # MediaPipe 설정 (손 제스처 전용)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     model_complexity=0,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils

# # HSV 색상 범위
# COLOR_RANGES = {
#     'red': ([0, 150, 100], [5, 255, 255], [175, 150, 100], [180, 255, 255]),
#     'orange': ([6, 150, 50], [20, 255, 255]),
#     'yellow': ([21, 150, 150], [35, 255, 255]),
#     'green':  ([40, 50, 50], [80, 255, 255]),
#     'blue':   ([90, 150, 50], [130, 255, 255]),
#     'purple': ([131, 150, 50], [160, 255, 255])
# }
# COLOR_CODE = {'red': 1, 'orange': 2, 'yellow': 3, 'green': 4, 'blue': 5, 'purple': 6}

# def detect_shape(contour):
#     epsilon = 0.04 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     vertices = len(approx)
#     if vertices == 3:
#         return 1  # 삼각형
#     elif vertices == 4:
#         (x, y, w, h) = cv2.boundingRect(approx)
#         aspect_ratio = w / float(h)
#         if 0.9 <= aspect_ratio <= 1.1:
#             return 2  # 정사각형
#         else:
#             return 3  # 직사각형
#     elif vertices == 5:
#         return 4  # 오각형
#     else:
#         area = cv2.contourArea(contour)
#         (_, _), radius = cv2.minEnclosingCircle(contour)
#         circle_area = np.pi * (radius ** 2)
#         if abs(1 - (area / circle_area)) < 0.2:
#             return 5  # 원
#     return 0  # 알 수 없는 도형

# def count_fingers(hand_landmarks, hand_label):
#     tips = [4, 8, 12, 16, 20]
#     fingers = []
#     if hand_label == "Left":
#         if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     else:
#         if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     for tip in tips[1:]:
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#     return sum(fingers)

# def create_mask(hsv_frame, color_range):
#     masks = []
#     for i in range(0, len(color_range), 2):
#         lower = np.array(color_range[i])
#         upper = np.array(color_range[i+1])
#         masks.append(cv2.inRange(hsv_frame, lower, upper))
#     combined_mask = masks[0]
#     for mask in masks[1:]:
#         combined_mask = cv2.bitwise_or(combined_mask, mask)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
#     processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
#     return processed_mask

# # 공유 변수와 락
# action_flag = threading.Event()
# action_type = None  # 'shape_count', 'color', 'shape_kind'
# lock = threading.Lock()

# def hand_gesture_thread():
#     global action_type
#     cap1 = cv2.VideoCapture(2)
#     cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     prev_action = None
#     while True:
#         ret, img = cap1.read()
#         if not ret:
#             print("카메라1 프레임 오류")
#             continue
#         img = cv2.flip(img, 1)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
#         finger_count = 0
#         if results.multi_hand_landmarks and results.multi_handedness:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 if not handedness.classification:
#                     continue
#                 hand_label = handedness.classification[0].label
#                 mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 finger_count = count_fingers(hand_landmarks, hand_label)
#                 y_pos = 50 if hand_label == "Left" else 100
#                 cv2.putText(img, f'{hand_label} Fingers: {finger_count}', (50, y_pos),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow("Hand Camera", img)
#         # 손가락 수에 따라 동작 결정
#         with lock:
#             if finger_count == 1 and prev_action != 1:
#                 action_type = 'shape_count'
#                 action_flag.set()
#                 prev_action = 1
#                 print("손가락 1개: 도형 개수 인식 요청")
#             elif finger_count == 2 and prev_action != 2:
#                 action_type = 'color'
#                 action_flag.set()
#                 prev_action = 2
#                 print("손가락 2개: 색상 인식 요청")
#             elif finger_count == 3 and prev_action != 3:
#                 action_type = 'shape_kind'
#                 action_flag.set()
#                 prev_action = 3
#                 print("손가락 3개: 도형 모양 인식 요청")
#             elif finger_count not in [1,2,3]:
#                 prev_action = None
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap1.release()

# def object_detection_thread():
#     cap2 = cv2.VideoCapture(1)
#     cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     while True:
#         action_flag.wait()
#         with lock:
#             current_action = action_type
#             action_flag.clear()
#         start_time = time.time()
#         detected_shapes = []
#         detected_colors = []
#         while time.time() - start_time < 2.0:
#             ret, img = cap2.read()
#             if not ret:
#                 continue
#             img = cv2.flip(img, 1)
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             for color_name, color_range in COLOR_RANGES.items():
#                 mask = create_mask(hsv, color_range)
#                 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 for cnt in contours:
#                     area = cv2.contourArea(cnt)
#                     if area < 1500:
#                         continue
#                     shape_num = detect_shape(cnt)
#                     if shape_num == 0:
#                         continue
#                     detected_shapes.append(shape_num)
#                     detected_colors.append(COLOR_CODE[color_name])
#                     x, y, w, h = cv2.boundingRect(cnt)
#                     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(img, f'{color_name} {shape_num}', (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imshow("Object Camera", img)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cap2.release()
#                 return
#         # 2초간 인식 결과 전송
#         try:
#             if current_action == 'shape_count':
#                 count_to_send = len(detected_shapes)
#                 client.write_register(135, count_to_send)
#                 print(f"[도형 개수] {count_to_send}개 → 레지스터 135")
#             elif current_action == 'color':
#                 if detected_colors:
#                     color_to_send = Counter(detected_colors).most_common(1)[0][0]
#                     client.write_register(130, color_to_send)
#                     print(f"[색상] 코드 {color_to_send} → 레지스터 130")
#                 else:
#                     client.write_register(130, 0)
#                     print("[색상] 없음: 0 전송")
#             elif current_action == 'shape_kind':
#                 if detected_shapes:
#                     shape_to_send = Counter(detected_shapes).most_common(1)[0][0]
#                     client.write_register(131, shape_to_send)
#                     print(f"[도형 모양] 코드 {shape_to_send} → 레지스터 131")
#                 else:
#                     client.write_register(131, 0)
#                     print("[도형 모양] 없음: 0 전송")
#         except Exception as e:
#             print(f"MODBUS 통신 오류: {e}")
#             client.close()
#             time.sleep(0.5)
#             client.connect()
#         time.sleep(0.5)  # 다음 인식까지 잠시 대기

# # 스레드 실행
# t1 = threading.Thread(target=hand_gesture_thread, daemon=True)
# t2 = threading.Thread(target=object_detection_thread, daemon=True)
# t1.start()
# t2.start()

# try:
#     while True:
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break
# except KeyboardInterrupt:
#     pass

# client.close()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from pymodbus.client import ModbusTcpClient
# from collections import Counter
# import time

# # MODBUS 설정
# client = ModbusTcpClient("192.168.213.78")   # 주소는 환경에 맞게 변경
# client.connect()

# # HSV 색상 범위 및 코드
# COLOR_RANGES = {
#     'red':    ([0, 150, 100], [5, 255, 255], [175, 150, 100], [180, 255, 255]),
#     'orange': ([6, 150, 50], [20, 255, 255]),
#     'yellow': ([21, 150, 150], [35, 255, 255]),
#     'green':  ([40, 50, 50], [80, 255, 255]),
#     'blue':   ([90, 150, 50], [130, 255, 255]),
#     'purple': ([131, 150, 50], [160, 255, 255])
# }
# COLOR_CODE = {'red': 1, 'orange': 2, 'yellow': 3, 'green': 4, 'blue': 5, 'purple': 6}
# SHAPE_CODE = {'circle': 1, 'triangle': 2, 'rectangle': 3, 'other': 0}

# def create_mask(hsv_frame, color_range):
#     masks = []
#     for i in range(0, len(color_range), 2):
#         lower = np.array(color_range[i])
#         upper = np.array(color_range[i+1])
#         masks.append(cv2.inRange(hsv_frame, lower, upper))
#     combined_mask = masks[0]
#     for mask in masks[1:]:
#         combined_mask = cv2.bitwise_or(combined_mask, mask)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
#     processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
#     return processed_mask

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# try:
#     while True:
#         ret, img = cap.read()
#         if not ret:
#             continue
#         img = cv2.flip(img, 1)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         detected_colors = []
#         detected_shapes = []
#         for color_name, color_range in COLOR_RANGES.items():
#             mask = create_mask(hsv, color_range)
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for cnt in contours:
#                 area = cv2.contourArea(cnt)
#                 if area < 1500:
#                     continue
#                 detected_colors.append(COLOR_CODE[color_name])
#                 # 도형 검출
#                 approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
#                 shape = 'other'
#                 if len(approx) == 3:
#                     shape = 'triangle'
#                 elif len(approx) == 4:
#                     shape = 'rectangle'
#                 elif len(approx) > 4:
#                     shape = 'circle'
#                 detected_shapes.append(SHAPE_CODE[shape])
#                 # 시각화
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(img, f'{color_name} {shape}', (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.imshow("Color & Shape Detection", img)

#         # 색상 결과 기록
#         if detected_colors:
#             color_to_send = Counter(detected_colors).most_common(1)[0][0]
#             client.write_register(130, color_to_send)
#             print(f"[색상] 코드 {color_to_send} → 레지스터 130")
#         else:
#             client.write_register(130, 0)
#             print("[색상] 없음: 0 전송")

#         # 도형 결과 기록
#         if detected_shapes:
#             shape_to_send = Counter(detected_shapes).most_common(1)[0][0]
#             client.write_register(131, shape_to_send)
#             print(f"[도형] 코드 {shape_to_send} → 레지스터 131")
#         else:
#             client.write_register(131, 0)
#             print("[도형] 없음: 0 전송")

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         time.sleep(0.5)
# finally:
#     cap.release()
#     client.close()
#     cv2.destroyAllWindows()
