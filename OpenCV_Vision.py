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