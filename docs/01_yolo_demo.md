## 1. 개발 환경
- NVIDIA GeForce GTX 1080 Ti 
- OS : 우분투 20.04 
- GPU 드라이버 : nvidia-driver-570 
- **CUDA Toolkit**: CUDA 12.4 
- **Python**: **Python 3.11** 
- **PyTorch**: **PyTorch 2.4+ (CUDA 12.4용)**
- Torchvision 버전 :   0.21.0+cu124
---
---

## 2. 실습 1 - 첫 추론
`src/yolo/predict_image.py`:
### 0). Conf 비교
- yolo26n 기준으로 conf : 0.1/0.25/0.5/0.8 비교

```python
from ultralytics import YOLO


confVal = [0.1,0.25,0.5,0.8]

for conf in confVal : 
	model = YOLO("yolo26n.pt")               # 처음 실행 시 가중치 자동 다운로드
	results = model("test.png", save=True,conf=conf)   # 결과가 runs/detect/predict/ 에 저장됨

	for box in results[0].boxes:
	    cls  = model.names[int(box.cls)]
	    conf = float(box.conf)
	    x1, y1, x2, y2 = box.xyxy[0].tolist()
	    print(f"{cls:12s} conf={conf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
```

#### 고찰
- Conf 값이 높은 값보다 낮은 값에 더 많은 사람이 분포하였다.
- 그 결과 Conf 기준이 높아질 수록 인식하는 객체가 줄어들었다.


#### test.png
![](lab1/img/test.png)
#### Conf=0.1, test1.jpg
![](lab1/img/0_test1.jpg)

#### Conf=0.25, test2.jpg
![](lab1/img/0_test2.jpg)
#### Conf=0.5, test3.jpg
![](lab1/img/0_test3.jpg)

#### Conf=0.8, test4.jpg
![](lab1/img/0_test4.jpg)

```

<1> 24명
person       conf=0.85 box=(87,191,111,232)
person       conf=0.84 box=(386,144,407,178)
person       conf=0.77 box=(237,152,251,189)
person       conf=0.57 box=(104,83,115,117)
person       conf=0.49 box=(0,89,12,120)
person       conf=0.47 box=(261,108,281,143)
person       conf=0.34 box=(242,62,258,91)
person       conf=0.33 box=(333,96,343,127)
person       conf=0.33 box=(58,39,68,68)
person       conf=0.32 box=(68,83,80,115)
person       conf=0.31 box=(228,47,240,74)
person       conf=0.30 box=(26,76,41,106)
person       conf=0.29 box=(140,6,152,25)
person       conf=0.28 box=(342,53,360,82)
person       conf=0.28 box=(296,82,320,110)
person       conf=0.27 box=(396,58,410,86)
person       conf=0.24 box=(196,68,208,97)
person       conf=0.21 box=(326,97,343,127)
person       conf=0.20 box=(242,62,259,91)
person       conf=0.18 box=(63,83,80,115)
person       conf=0.16 box=(98,84,115,117)
person       conf=0.16 box=(304,82,320,111)
person       conf=0.15 box=(64,83,80,116)
person       conf=0.14 box=(267,110,280,142)
person       conf=0.12 box=(396,57,410,86)

<2> 16명
person       conf=0.85 box=(87,191,111,232)
person       conf=0.84 box=(386,144,407,178)
person       conf=0.77 box=(237,152,251,189)
person       conf=0.57 box=(104,83,115,117)
person       conf=0.49 box=(0,89,12,120)
person       conf=0.47 box=(261,108,281,143)
person       conf=0.34 box=(242,62,258,91)
person       conf=0.33 box=(333,96,343,127)
person       conf=0.33 box=(58,39,68,68)
person       conf=0.32 box=(68,83,80,115)
person       conf=0.31 box=(228,47,240,74)
person       conf=0.30 box=(26,76,41,106)
person       conf=0.29 box=(140,6,152,25)
person       conf=0.28 box=(342,53,360,82)
person       conf=0.28 box=(296,82,320,110)
person       conf=0.27 box=(396,58,410,86)

<3> 4명
person       conf=0.85 box=(87,191,111,232)
person       conf=0.84 box=(386,144,407,178)
person       conf=0.77 box=(237,152,251,189)
person       conf=0.57 box=(104,83,115,117)

<4> 2명
person       conf=0.85 box=(87,191,111,232)
person       conf=0.84 box=(386,144,407,178)
```
---


### 1). 모델 비교
- conf = 0.5,imgsz = 640, device=0 (Cuda) 기준으로  비교
- yolo26n, yolo26s, yolo26m
``` python
from ultralytics import YOLO


confVal = [0.1,0.25,0.5,0.8]
imgVal = [320,640, 1280]
modelVal = ["yolo26n", "yolo26s", "yolo26m"]
deviceVal = [0,"CPU"]

for val in range(3) : 
	model = YOLO(modelVal[val])               # 처음 실행 시 가중치 자동 다운로드
	results = model("test.png",device=deviceVal[0], imgsz=imgVal[1], save=True,conf=confVal[2])   # 결과가 runs/detect/predict/ 에 저장됨

	for box in results[0].boxes:
	    cls  = model.names[int(box.cls)]
	    conf = float(box.conf)
	    x1, y1, x2, y2 = box.xyxy[0].tolist()
	    print(f"{cls:12s} conf={conf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

```
#### 고찰
- 각 모델 마다 모델 크기의 차이가 있다.
	- yolo26n : 가장 작고 빠름
	- yolo26s : 속도와 정확도의 균형
	- yolo26m : 더 정확하지만 느리다.
- 모델이 커질 수록 모델 내부에 합성곱 층과 채널 수가 증가하므로, 더 많은 특징을 학습하고 처리할 수 있어서 작은 객체를 더 잘 인식한다.

#### yolo26n

![](lab1/img/1_test1.jpg)

####  yolo26s

![](lab1/img/1_test2.jpg)

####  yolo26m
- 가장 작은 Sports Ball 까지 인식한다.
![](lab1/img/1_test3.jpg)

```
predict1 4명
person       conf=0.85 box=(87,191,111,232)
person       conf=0.84 box=(386,144,407,178)
person       conf=0.77 box=(237,152,251,189)
person       conf=0.57 box=(104,83,115,117)

predict2 15명
person       conf=0.84 box=(386,144,407,179)
person       conf=0.83 box=(243,62,258,91)
person       conf=0.80 box=(87,191,111,232)
person       conf=0.77 box=(235,151,251,189)
person       conf=0.77 box=(296,82,320,112)
person       conf=0.75 box=(261,108,281,142)
person       conf=0.70 box=(187,68,209,98)
person       conf=0.68 box=(57,40,68,67)
person       conf=0.66 box=(396,58,411,86)
person       conf=0.61 box=(66,84,80,116)
person       conf=0.60 box=(228,48,240,75)
person       conf=0.60 box=(0,88,12,120)
person       conf=0.56 box=(332,96,343,127)
person       conf=0.54 box=(327,96,343,127)
person       conf=0.51 box=(105,84,115,117)


predict3 12명
person       conf=0.77 box=(296,82,321,112)
person       conf=0.76 box=(386,143,407,179)
person       conf=0.73 box=(0,86,12,120)
person       conf=0.71 box=(243,62,258,91)
person       conf=0.70 box=(327,96,343,128)
person       conf=0.69 box=(187,67,210,98)
person       conf=0.68 box=(87,191,111,232)
person       conf=0.68 box=(65,84,80,117)
person       conf=0.68 box=(262,108,281,143)
person       conf=0.66 box=(395,58,411,87)
person       conf=0.66 box=(342,52,360,83)
person       conf=0.64 box=(25,74,41,105)
person       conf=0.61 box=(234,151,251,189)
sports ball  conf=0.50 box=(258,87,264,92)
```


### 2). 이미지 크기 비교
- conf = 0.5, yolo26n, device=0 (Cuda) 기준으로  비교
- imgsz =320 / 640 / 1280 비교

``` python
from ultralytics import YOLO


confVal = [0.1,0.25,0.5,0.8]
imgVal = [320,640, 1280]
modelVal = ["yolo26n", "yolo26s", "yolo26m"]
deviceVal = [0,"CPU"]

for val in range(3) : 
	model = YOLO(modelVal[0])               # 처음 실행 시 가중치 자동 다운로드
	results = model("test.png",device=deviceVal[0], imgsz=imgVal[val], save=True,conf=confVal[2])   # 결과가 runs/detect/predict/ 에 저장됨

	for box in results[0].boxes:
	    cls  = model.names[int(box.cls)]
	    conf = float(box.conf)
	    x1, y1, x2, y2 = box.xyxy[0].tolist()
	    print(f"{cls:12s} conf={conf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

```
#### 고찰 
- 입력되는 이미지의 크기가 커질 수록 작은 물체에 더 많은 픽셀이 할당되므로, 큰 입력 이미지 크기를 설정할 수록 작은 객체를 찾는데 유리하다.

#### imgsz = 320

![](lab1/img/2_test1.jpg)
#### imgsz = 640

![](lab1/img/2_test2.jpg)
#### imgsz = 1280

![](lab1/img/2_test3.jpg)

```

predict 3명
person       conf=0.65 box=(387,145,407,177)
person       conf=0.64 box=(237,151,251,189)
person       conf=0.56 box=(88,192,108,228)

predict-2 4명
person       conf=0.85 box=(87,191,111,232)
person       conf=0.84 box=(386,144,407,178)
person       conf=0.77 box=(237,152,251,189)
person       conf=0.57 box=(104,83,115,117)

predict-3 13명
person       conf=0.86 box=(386,143,407,178)
person       conf=0.85 box=(242,62,258,91)
person       conf=0.83 box=(296,81,320,111)
person       conf=0.80 box=(0,87,13,120)
person       conf=0.79 box=(195,67,210,98)
person       conf=0.77 box=(228,47,240,75)
person       conf=0.76 box=(234,151,251,188)
person       conf=0.74 box=(104,84,115,117)
person       conf=0.69 box=(395,58,411,86)
person       conf=0.59 box=(342,53,360,83)
person       conf=0.58 box=(25,74,41,106)
person       conf=0.54 box=(328,96,343,127)
person       conf=0.51 box=(141,6,152,25)


```

---
### 3). CPU vs CUDA
``` python
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

```

#### 고찰
- 검출한 사람의 수와 Conf는 동일하다. 
- 다만, CUDA가 12배 이상 빠른 속도로 추론을 마무리 지었다.

#### CPU

![](lab1/img/3_test1.jpg)

#### CUDA

![](lab1/img/3_test2.jpg)


```

<CUDA>
15 persons, 7.1ms.
Speed: 13.1ms preprocess, 7.1ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)

person       conf=0.84 box=(386,144,407,179)
person       conf=0.83 box=(243,62,258,91)
person       conf=0.80 box=(87,191,111,232)
person       conf=0.77 box=(235,151,251,189)
person       conf=0.77 box=(296,82,320,112)
person       conf=0.75 box=(261,108,281,142)
person       conf=0.70 box=(187,68,209,98)
person       conf=0.68 box=(57,40,68,67)
person       conf=0.66 box=(396,58,411,86)
person       conf=0.61 box=(66,84,80,116)
person       conf=0.60 box=(228,48,240,75)
person       conf=0.60 box=(0,88,12,120)
person       conf=0.56 box=(332,96,343,127)
person       conf=0.54 box=(327,96,343,127)
person       conf=0.51 box=(105,84,115,117)

<CPU>
15 persons, 86.5ms
Speed: 1.0ms preprocess, 86.5ms inference, 0.2ms postprocess per image at shape (1, 3, 384, 640)

person       conf=0.84 box=(386,144,407,179)
person       conf=0.83 box=(243,62,258,91)
person       conf=0.80 box=(87,191,111,232)
person       conf=0.77 box=(235,151,251,189)
person       conf=0.77 box=(296,82,320,112)
person       conf=0.75 box=(261,108,281,142)
person       conf=0.70 box=(187,68,209,98)
person       conf=0.68 box=(57,40,68,67)
person       conf=0.66 box=(396,58,411,86)
person       conf=0.61 box=(66,84,80,116)
person       conf=0.60 box=(228,48,240,75)
person       conf=0.60 box=(0,88,12,120)
person       conf=0.56 box=(332,96,343,127)
person       conf=0.54 box=(327,96,343,127)
person       conf=0.51 box=(105,84,115,117)


```

---

### 4). 예외 : 틀린 사례

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")               # 처음 실행 시 가중치 자동 다운로드
for i in range(1,9,1):

	path = f"except/except{i}.jpg"
	results = model(path, save=True)   # 결과가 runs/detect/predict/ 에 저장됨

	for box in results[0].boxes:
	    cls  = model.names[int(box.cls)]
	    conf = float(box.conf)
	    x1, y1, x2, y2 = box.xyxy[0].tolist()
	    print(f"{cls:12s} conf={conf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
	    
	    
```

#### except1.jpg
 - 컵과 유사한 모양을 보고 cup이라 판단하였으나 재질을 파악했을 때, 컵이 아니라 휴지라는 것을 알 수 있다.
 - 또한, 투명도를 관찰하였을 때, 이것이 반사된 것인지 실제 사물인지 분간 할 수 있을 것.
```
tv           conf=0.77 box=(655,0,1279,156)
tv           conf=0.72 box=(0,1,437,411)
cup          conf=0.62 box=(601,493,740,657)
cup          conf=0.61 box=(847,420,1007,568)
book         conf=0.28 box=(8,594,809,1278)
cup          conf=0.27 box=(601,493,741,657)

```


![](img/except1.jpg)
#### except2.jpg
- 색깔과 형태를 보고 유사한 것을 찾기는 하였으나, 학습되지 않는 사물이라 파악하지 못한 것이라 판단된다.
```
teddy bear   conf=0.78 box=(81,215,604,988)
carrot       conf=0.31 box=(500,379,618,708)

```


![](img/except2.jpg)

#### except5.jpg
- 가려져 있거나 형태를 파악할 수 없는 상황에서는 사물을 인지조차 하지 못하는 것 같다.
```
person       conf=0.89 box=(396,656,719,1110)
person       conf=0.83 box=(64,455,213,721)
couch        conf=0.81 box=(0,574,226,1078)
dining table conf=0.38 box=(258,859,551,1279)
```

![](img/except5.jpg)


---
---
## 3. 실습2 - 웹캠 실시간 데모

`src/yolo/webcam_demo.py`:

```python
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
```

![](lab1/img/YOLO_webcam.png)




## 4.실습 3 - 영상 파일 + FPS 측정

``` python

import time
from ultralytics import YOLO

# 조건 리스트  
modelVal = ["yolo26n", "yolo26s", "yolo26m"]
confVal = [0.1,0.25,0.5,0.8]
imgVal = [320,640, 1280]
deviceVal = [0,"CPU"]

  
# i값을 조건 리스트 인덱스에 넣어 측정
for i in range(3) :
	model = YOLO(modelVal[i])
	t0, n = time.time(), 0
	for result in model.predict(source="football.mp4", stream=True, conf=confVal[2], imgsz = imgVal[1], device= deviceVal[0] ,verbose =False, show=True, save = True ):
		n += 1
		persons = [b for b in result.boxes if model.names[int(b.cls)] == "person"]
		print(f"frame {n:4d}: person {len(persons)}명")
		
	print(f"평균 {n / (time.time() - t0):.1f} FPS")
```


![](lab1/img/video.png)

### 1). conf
- conf 0.1/0.25/0.5/ 0.8 비교
- yolo26n, imgsz = 640, device = CUDA 고정
- FPS는 conf가 커질수록 커지고, 프레임당 평균 감지 수는 줄어드는 양상을 보인다. 
```
<conf 0.1>
56FPS ( 879프레임 처리)
총 검출된 사람수 13,831명
프레임당 평균 감지 수: 약 15.73명
최대 감지 32명 

<conf 0.25>
63.4FPS ( 879프레임 처리)
총 검출된 사람수 8,755명
프레임당 평균 감지 수: 약 9.96명
최대 감지 20명 


<conf 0.5>
66FPS ( 879프레임 처리)
총 검출된 사람수 4,383명
프레임당 평균 감지 수: 약 4.99명
최대 감지 13명 


<conf 0.8>
70.1FPS ( 879프레임 처리)
총 검출된 사람수 137명
프레임당 평균 감지 수: 약 0.156명
최대 감지 3명 

```
### 2). 모델
- yolo26n, yolo26s, yolo26m 비교
- conf = 0.5, imgsz = 640, device = CUDA
- 모델이 커질수록 처리 속도는 느려지지만 정확도는 올라간다는 것을 잘 보여준다.
```
<yolo26n>
62FPS ( 879프레임 처리)
총 검출된 사람수 4,383명
프레임당 평균 감지 수: 약 4.99명
최대 감지 13명 

<yolo26s>
65.8FPS ( 879프레임 처리)
총 검출된 사람수 7,260명
프레임당 평균 감지 수: 약 8.26명
최대 감지 16명 

<yolo26m>
50.3FPS ( 879프레임 처리)
총 검출된 사람수 7,688명
프레임당 평균 감지 수: 약 8.75명
최대 감지 18명 

```

### 3). 입력 이미지 크기
- imgsz =320 / 640 / 1280 비교
- yolo26n, conf = 0.5, device = CUDA 고정
- 입력 이미지 크기가 커질 수록 검출된 사람의 수가 증가한다.
- 640 사이즈에서 1280 사이즈로 커질 때, 픽셀 수는 4배 증가하므로 FPS 속도가 감소한다.
- 다만, 320 사이즈와 640 사이즈 같은 저해상도에서는 GPU 연산 특성에 의해 FPS 차이가 거의 없거나 오히려 저해상도에서 FPS가 더 높은 현상이 자주 관찰된다.

```

<imgsz = 320 > 
FPS : 62
총 검출된 사람수 : 562명 
프레임당 평균 감지수 : 0.64명
최대 감지 수 : 3

<imgsz = 640 > 
FPS : 66.3
총 검출된 사람수 : 3,959명 
프레임당 평균 감지수 : 4.5명
최대 감지 수 : 13명

<imgsz = 1280 > 
FPS : 48.2
총 검출된 사람수 : 5,335명
프레임당 평균 감지수 : 6.07명 
최대 감지 수 : 18명



```


### 4). device
- CUDA Vs CPU
- imgsz=640,yolo26n, conf = 0.5, device = CUDA 고정
- 처리 능력 보다는 처리 속도에서 차이를 보인다. 
```
< CUDA > 
FPS : 58.7FPS
총 검출된 사람수 : 3,959명 
프레임당 평균 감지수 : 4.5명
최대 감지 수 : 13명


< CPU > 
FPS : 19.9
총 검출된 사람수 : 4,204명 
프레임당 평균 감지수 : 4.78명
최대 감지 수 : 13명


```



