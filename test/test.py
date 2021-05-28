import cv2
import numpy as np

while True:
    img = cv2.imread('./5_1.jpg')
        
        #YCrCb 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR 값을 HSV 값으로 변경

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

    dst = cv2.cvtColor(hand, cv2.COLOR_GRAY2BGR)
    contours, hierachy=cv2.findContours(hand, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
    points1 = []
    result_cx = []
    result_cy = []
    
    # 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
            max = area
            maxcnt = cnt
    cv2.drawContours(dst, [maxcnt], 0, (0, 0, 255), 2)

    M = cv2.moments(maxcnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # set values as what you need in the situation
        cX, cY = 0, 0
    
    approx = cv2.approxPolyDP(maxcnt,0.02*cv2.arcLength(maxcnt,True),True)
    #cv2.drawContours(dst, [approx], 0, (0,255,0), 3)
    hull = cv2.convexHull(approx)
    for point in hull:
        if cy > point[0][1]:
            # print('hull: ', hull)
            # print('point[0][1]: ', point[0][1])
            # print('cy: ', cy)
            points1.append(tuple(point[0])) 
    cv2.drawContours(dst, [maxcnt], 0, (0, 0, 255), 2)
    # for point in points1:
    #     print('cy: ', cy)
    #     print('point: ',point)
    #     cv2.circle(dst, tuple(point), 10, [255, 0, 0], -1)
    cv2.circle(dst, (156, 156), 10, [255, 0, 0], -1) 
            
    cv2.imshow('dst', dst)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if True:
    #     break