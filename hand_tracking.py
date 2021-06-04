import numpy as np
import cv2

##함수 넣으세요
def pt(frame):
    ### ycrcb <- HSV로 대체. HSV 정확도가 더 낫다고 판단.
    # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # mask_hand = cv2.inRange(ycrcb,np.array([0,133,77]),np.array([255,173,127]))
 
    # # cv2.imshow("Hands",mask_hand)
    # # if cv2.waitKey(1) & 0xFF == ord('q'):
    # #     break

    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # # 침식 연산 적용 ---②
    # erosion = cv2.erode(mask_hand, k)
    ###
    frame=frame[100:600, 200:400]
    ### HSV방법으로 색공간 변경
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BGR 값을 HSV 값으로 변경

    ##테스트 필요
    hue = np.array([0, 48, 80])
    hue2 = np.array([20, 255, 255]) #손 색깔

    hand = cv2.inRange(hsv, hue, hue2) #손 색깔 범위 정의

    #테스트 필요 (3, 3) 조정
    hand = cv2.GaussianBlur(hand, (3,3), 0) #가우시안 블러링 -> 노이즈 제거
    
    #Morphology -> 노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, k)
    hand = cv2.erode(hand, k)

    dst = cv2.cvtColor(hand, cv2.COLOR_GRAY2BGR) # hand 객체를 BGR 회색조 이미지로 변환하여 dst에 저장 -> 컨투어를 검출하는 주된 요소는 하얀색 객체를 검출. 
    
    ###이진화
    # dst = cv2.adaptiveThreshold(hand, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    ###
    
    return hand, dst