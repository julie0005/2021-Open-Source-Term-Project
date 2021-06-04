import numpy as np
import math
import cv2

##함수 넣으세요
def pt(hand, dst):
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
    contours, hierarchy = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #찾은 윤곽선 중 가장 큰 영역 찾기
    max = 0
    cntt=0
    save=0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
            max = area
            maxcnt = cnt
            save=cntt
        cntt+=1
    
    

    #cv2.drawContours(dst, [maxcnt], -1, (0, 0, 255), 2) #가장 큰 영역의 윤곽선 그려주기
    
    
    stencil = np.zeros(dst.shape).astype(dst.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [maxcnt], color)
    

    dst=stencil

    # contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # #찾은 윤곽선 중 가장 큰 영역 찾기
    # max = 0
    # cntt=0
    # save=0
    # maxcnt = None
    # for cnt in contours :
    #     area = cv2.contourArea(cnt)
    #     if(max < area) :
    #         max = area
    #         maxcnt = cnt
    #         save=cntt
    #     cntt+=1
    
    
    
   
    #dst = cv2.cvtColor(hand, cv2.COLOR_GRAY2BGR)

    points1 = []
    result_cx = []
    result_cy = []
        

    M = cv2.moments(maxcnt) #윤곽선에서 모멘트(중심점) 계산
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
    check=0

    cy+=15
    #중심점(cy)보다 높이 있는 볼록점의 y좌표(point[0][1])들은 손가락 끝 부분(points1) 리스트에 추가.
    for point in hull:
        if cy < point[0][1]:
             check=point[0][1]-cy
        if cy > point[0][1]:
             leng=math.sqrt((cx-point[0][0])**2+(cy-point[0][1])**2)
             if leng>check+10:
                points1.append(tuple(point[0]))
    

    cv2.circle(dst, tuple((cx,cy)), 10, [0,255,0],-1)    

    #points1(손가락 끝 부분)에 속한 요소들에 circle 표시
    for point in points1:
        cv2.circle(dst, tuple(point), 10, [255, 0, 0], -1)

    finger_count = len(points1) #손가락 개수 counting
    
    cv2.imshow('dst', dst) #결과를 담은 영상 출력

    return finger_count