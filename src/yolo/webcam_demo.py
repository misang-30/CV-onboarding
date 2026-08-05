import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)          # 웹캠이 여러 개면 1, 2로 바꿔보기

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()   # 박스가 그려진 이미지 반환

    cv2.imshow("YOLO webcam (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
