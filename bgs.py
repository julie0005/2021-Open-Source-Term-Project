import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
print(cap.get(3), cap.get(4))
ret = cap.set(3,320)
ret = cap.set(4,240)
while(True):
    ret, frame = cap.read() 


    # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # mask_hand = cv2.inRange(ycrcb,np.array([0,133,77]),np.array([255,173,127]))
 
    # # cv2.imshow("Hands",mask_hand)
    # # if cv2.waitKey(1) & 0xFF == ord('q'):
    # #     break

    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # # 침식 연산 적용 ---②
    # erosion = cv2.erode(mask_hand, k)

    laptime = time.time()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    hue = np.array([0, 48, 80])
    hue2 = np.array([20, 255, 255])#손 색깔
    hand = cv2.inRange(hsv, hue, hue2)

    hand = cv2.GaussianBlur(hand, (3,3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, k)
    hand = cv2.erode(hand, k)
    dst = cv2.cvtColor(hand, cv2.COLOR_GRAY2BGR)

    #LABELING
    # _, src_bin = cv2.threshold(hand, 0, 255, cv2.THRESH_OTSU)

    # cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)


    # for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
    #     (x, y, w, h, area) = stats[i]

    #     # 노이즈 제거
    #     if area < 20:
    #         continue

    #     cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))

    # 가장 큰 영역 박스 생성
    # (x, y, w, h, area) = stats[1]
    # if area < 20:
    #     continue
    # cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))

    #CONTOUR
    #경계선 찾음
    contours, hierarchy = cv2.findContours(hand, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
            max = area
            maxcnt = cnt
    cv2.drawContours(dst, [maxcnt], 0, (0, 0, 255), 2)

    # for i in contours:
    #     #hull = cv2.convexHull(i,clockwise=True)
    #     cv2.drawContours(dst, [i], 0, (0,0,255),2)

    contours, hierachy=cv2.findContours(hand, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
    points1 = []
    result_cx = []
    result_cy = []
        
    for i in contours:
        M = cv2.moments(i)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # set values as what you need in the situation
            cX, cY = 0, 0

    # 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
            max = area
            maxcnt = cnt
    
    approx = cv2.approxPolyDP(maxcnt,0.02*cv2.arcLength(maxcnt,True),True)
    #cv2.drawContours(dst, [approx], 0, (0,255,0), 3)
    hull = cv2.convexHull(approx)
    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0])) 

    for point in points1:
        cv2.circle(dst, tuple(point), 10, [255, 0, 0], -1)
            
            
    cv2.imshow('dst', dst)


    # cv2.imshow('src', hand)
    # cv2.imshow('src_bin', src_bin)
    # cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cv2.imshow('Erode', hand)
    # 결과 출력
    # merged = np.hstack((mask_hand, erosion))
    # cv2.imshow('Erode', merged)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()