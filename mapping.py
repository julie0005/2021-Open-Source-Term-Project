import numpy as np
import cv2
import time
import webbrowser

##손가락 개수에 따른 바로가기
def pt(cnt, prev_finger_count, finger_count):
    if cnt == 0:
        prev_finger_count = finger_count
        return cnt+1, prev_finger_count, 0
    elif cnt < 50:
        if prev_finger_count == finger_count:
            return cnt+1, prev_finger_count, 0
        else:
            prev_finger_count = finger_count
            return 0, prev_finger_count, 0
    
    if cnt == 50:
        if finger_count == 5:
            return 0, 0, 1
        elif finger_count == 4:
            webbrowser.open("https://www.naver.com/")
            time.sleep(5)
            return 0, 0, 0
        elif finger_count == 3:
            webbrowser.open("https://www.google.com/")
            time.sleep(5)
            return 0, 0, 0
        elif finger_count == 2:
            webbrowser.open("https://www.ajou.ac.kr/")
            time.sleep(5)
            return 0, 0, 0
        elif finger_count == 1:
            webbrowser.open("https://github.com/")
            time.sleep(5)
            return 0, 0, 0
        else:
            return 0, 0, 0