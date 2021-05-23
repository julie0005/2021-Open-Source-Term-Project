import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0) #카메라를 VideoCapture 타입의 객체로 반환(영상)
print(cap.get(3), cap.get(4))

#가로 세로의 크기: 320x240
ret = cap.set(3,320)
ret = cap.set(4,240)

while(True):
    ret, frame = cap.read() #카메라로부터 현재 영상 하나를 읽어옴

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

    laptime = time.time()

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

    ###LABELING -> 박스로 추출. Contour에 비해 정확도 떨어짐
    # _, src_bin = cv2.threshold(hand, 0, 255, cv2.THRESH_OTSU)
    # cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

    # for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
    #     (x, y, w, h, area) = stats[i]

    #     # 노이즈 제거
    #     if area < 20:
    #         continue

    #     cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))

    # #가장 큰 영역 박스 생성
    # (x, y, w, h, area) = stats[1]
    # if area < 20:
    #     continue
    # cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))
    # cv2.imshow('dst', dst)
    ###

    ###CONTOUR
    #윤곽선 찾음
    ##테스트 필요 - 파라미터 조정
    contours, hierarchy = cv2.findContours(hand, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #찾은 윤곽선 중 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
            max = area
            maxcnt = cnt
    cv2.drawContours(dst, [maxcnt], 0, (0, 0, 255), 2) #가장 큰 영역의 윤곽선 그려주기

    points1 = []
    result_cx = []
    result_cy = []
        
    for i in contours:
        M = cv2.moments(i) #윤곽선에서 모멘트(중심점) 계산
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) #중심점의 x좌표
            cy = int(M["m01"] / M["m00"]) #중심점의 y좌표
        else:
            # set values as what you need in the situation
            cX, cY = 0, 0

    #윤곽선 근사화(단순화)
    approx = cv2.approxPolyDP(maxcnt,0.02*cv2.arcLength(maxcnt,True),True)

    #approx 윤곽선에서 볼록 껍질(볼록점)을 검출
    hull = cv2.convexHull(approx)

    #중심점(cy)보다 높이 있는 볼록점의 y좌표(point[0][1])들은 손가락 끝 부분(points1) 리스트에 추가.
    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0])) 

    #points1(손가락 끝 부분)에 속한 요소들에 circle 표시
    for point in points1:
        cv2.circle(dst, tuple(point), 10, [255, 0, 0], -1)

    #print(len(points1)) #손가락 개수 counting
    
    cv2.imshow('dst', dst) #결과를 담은 영상 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()