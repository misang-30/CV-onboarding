from ultralytics import YOLO


confVal = [0.1,0.25,0.5,0.8]
imgVal = [320,640, 1280]
modelVal = ["yolo26n", "yolo26s", "yolo26m"]
deviceVal = [0,"CPU"]

for val in range(2) : 
	model = YOLO(modelVal[1])               # 처음 실행 시 가중치 자동 다운로드
	results = model("test.png",device=deviceVal[val], imgsz=imgVal[1], save=True,conf=confVal[2])   # 결과가 runs/detect/predict/ 에 저장됨

	for box in results[0].boxes:
	    cls  = model.names[int(box.cls)]
	    conf = float(box.conf)
	    x1, y1, x2, y2 = box.xyxy[0].tolist()
	    print(f"{cls:12s} conf={conf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
