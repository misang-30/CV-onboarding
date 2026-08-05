import time
from ultralytics import YOLO

# 조건 리스트  
modelVal = ["yolo26n", "yolo26s", "yolo26m"]
confVal = [0.1,0.25,0.5,0.8]
imgVal = [320,640, 1280]
deviceVal = [0,"CPU"]

  
# i값을 조건 리스트 인덱스에 넣어 측정
for i in range(2) :
	model = YOLO(modelVal[0])
	t0, n = time.time(), 0
	for result in model.predict(source="football.mp4", stream=True, conf=confVal[2], imgsz = imgVal[1], device= deviceVal[i] ,verbose =False, show=True, save = True ):
		n += 1
		persons = [b for b in result.boxes if model.names[int(b.cls)] == "person"]
		print(f"frame {n:4d}: person {len(persons)}명")
		
	print(f"평균 {n / (time.time() - t0):.1f} FPS")
	
