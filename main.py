import numpy as np
import cv2
import filter as ft
import finger_tracking as ft
import mapping as mp

classes = ["Hand"]
def yolo(frame, score_threshold, nms_threshold):
    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet("./yolov4-tiny2_best.weights", "./yolov4-tiny2.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)
    
    # 후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')
    print("\n\n============================== classes ==============================")
    if len(indexes)==2:
        index=indexes[1][0]
        bindex=indexes[0][0]
        height=-1
        maxi=index if boxes[bindex][2]<boxes[index][2] else bindex
        mini=bindex if boxes[bindex][2]>boxes[index][2] else index

        if boxes[mini][2]/boxes[maxi][2]>0.38:
            index=index if (boxes[bindex][3]<boxes[index][3]) else bindex
            x, y, w, h = boxes[height]

            class_name = classes[class_ids[index]]
            label = f"{class_name} {confidences[index]:.2f}"
            color = colors[class_ids[index]]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            xi=x-30
            yi=y-50
            wi=w+40
            hi=h+50
            
            xend=480 if xi+wi>480 else xi+wi
            yend=640 if yi+hi>640 else yi+hi
            xstart= 0 if xi<0 or xend==480 else xi
            ystart= 0 if yi<0 or yend==640 else yi

            
            frame2=frame[ystart:yend, xstart:xend]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            
            # 탐지된 객체의 정보 출력
            print(f"[{class_name}({index})] conf: {confidences[index]} / x: {x} / y: {y} / width: {w} / height: {h}")
            return frame,frame2

    elif len(indexes)==1:
        index=indexes[0][0]
        x, y, w, h = boxes[index]
        class_name = classes[class_ids[index]]
        label = f"{class_name} {confidences[index]:.2f}"
        color = colors[class_ids[index]]

        # 사각형 테두리 그리기 및 텍스트 쓰기
        xi=x-30
        yi=y-50
        wi=w+40
        hi=h+50
        
        xend=480 if xi+wi>480 else xi+wi
        yend=640 if yi+hi>640 else yi+hi
        xstart= 0 if xi<0 or xend==480 else xi
        ystart= 0 if yi<0 or yend==640 else yi
        
        frame2=frame[ystart:yend, xstart:xend]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        
        # 탐지된 객체의 정보 출력
        print(f"[{class_name}({index})] conf: {confidences[index]} / x: {x} / y: {y} / width: {w} / height: {h}")
        return frame,frame2
    else:
        print("손이 인식되지 않거나 세 개 이상입니다.")
    
    frame2=[]
    return frame,frame2

cap = cv2.VideoCapture(0) #카메라를 VideoCapture 타입의 객체로 반환(영상)
print(cap.get(3), cap.get(4))

#가로 세로의 크기: 320x240
ret = cap.set(3,640)
ret = cap.set(4,480)

while(True):
    ret, frame = cap.read() #카메라로부터 현재 영상 하나를 읽어옴
    
    #hand_tracking(승은) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame,frame2=yolo(frame=frame,score_threshold=0.4,nms_threshold=0.5)
    cv2.imshow('original',frame)
    if len(frame2)==0:
        print(len(frame2))
        continue
    
    hand, dst = ft.pt(frame2) #-> 함수 이름 적당한 걸로 바꾸기
    
    #finger_tracking(은주)
    finger_count = ft.pt(hand, dst) #-> 함수 이름 적당한 걸로 바꾸기

    # print(finger_count) # 손가락 개수 출력

    #mapping(윤정)
    mp.pt(finger_count) #-> 함수 이름 적당한 걸로 바꾸기
    
    

cap.release()
cv2.destroyAllWindows()