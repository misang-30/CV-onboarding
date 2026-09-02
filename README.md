# CV-onboarding  
- 컴퓨터비전 연구실 학부연구생 온보딩 과정 기록입니다.

## 환경
- Ubuntu 20.04 LTS / GTX 1080Ti / CUDA 12.4 / PyTorch 2.4+

## 목차
1. [우분투 설치](docs/01_ubuntu_install.md)
2. [NVIDIA 드라이버 · CUDA 설치](docs/02_nvidia_cuda.md)
3. [VS Code 개발 환경 설정](docs/03_vscode_setup.md)
4. [Pytorch 설치](docs/04_pytorch.md)
5. [YOLO 데모 및 실험](docs/04_yolo_demo.md)
6. [SmallCNN 학습 파이프라인 구축](docs/05_training_basic.md)
7. [SmallCNN 실험 노트](docs/06_experiments.md)

## 실행 방법
```bash
conda env create -f environment.yml
conda activate pytorch_env
python src/yolo/webcam_demo.py
```



## Lab
### 1). YOLO
```
cv-onboarding/
├── README.md                  
├── .gitignore
├── environment.yml            # conda 환경 기록
├── docs/
│   ├── 01-ubuntu-install.md   # 우분투 설치 기록
│   ├── 02-nvidia-cuda.md      # 드라이버·CUDA 설치 기록
│   ├── 03-vscode-setup.md     # VS Code 설정 기록
│   └── 04-yolo-demo.md        # YOLO 실습 기록
├── img/                       # 문서에 넣을 캡처 이미지
└── src/
    ├── check_env.py
    └── yolo/
        ├── predict_image.py
        ├── predict_video.py
        └── webcam_demo.py
```


### 2).SmallCNN Model Train
```
cv-onboarding/
├── docs/
│   ├── ...
│   ├── 05-training-basics.md     # 문서 1: 학습 파이프라인 구축 기록
│   └── 06-experiments.md         # 문서 2: 실험 노트 (핵심 산출물)
└── src/
    └── train/                    # 정리된 코드
        ├── data.py               # Day1 : 데이터 불러오기. Dataset / DataLoader / split
        ├── model.py              # Day2 : 모델 직접 구현하기 . 직접 구현한 CNN
        ├── train.py              # Day2 : 학습 루프 구현하기 / Day4 : 변수 통제 실험.
        ├── plot_curves.py        # CSV → 곡선 그리기 (Day 1~2용)
        ├── overfit.py            # Day3 : 과적합 제조하기
        ├── dataleak.py           # Day5 - Lab B : 데이터 누수 실험 
        ├── labC.py               # Day5 - Lab C : Val set 오염 실험.
        ├── labD.py               # Day5 - Lab D : SmallCNN , ResNet18 비교 실험.      
        └── configs/
            └── baseline.yaml
        └── old/                           # 결과물만 확인한 코드
            ├── data_prev.py               # Day1 : 데이터 불러오기
            ├── model_prev.py              # Day2 : 모델 직접 구현하기 
            └── train_prev.py              # Day3 : 학습 루프 구현하기 

        
```



