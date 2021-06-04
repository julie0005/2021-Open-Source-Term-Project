import numpy as np
import cv2
import hand_tracking as ht
import finger_tracking as ft
import mapping as mp

cap = cv2.VideoCapture(0) #카메라를 VideoCapture 타입의 객체로 반환(영상)
print(cap.get(3), cap.get(4))

#가로 세로의 크기: 320x240
ret = cap.set(3,640)
ret = cap.set(4,480)

while(True):
    ret, frame = cap.read() #카메라로부터 현재 영상 하나를 읽어옴

    #hand_tracking(승은) 
    hand, dst = ht.pt(frame) #-> 함수 이름 적당한 걸로 바꾸기
    
    #finger_tracking(은주)
    finger_count = ft.pt(hand, dst) #-> 함수 이름 적당한 걸로 바꾸기

    # print(finger_count) # 손가락 개수 출력

    #mapping(윤정)
    mp.pt(finger_count) #-> 함수 이름 적당한 걸로 바꾸기

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()