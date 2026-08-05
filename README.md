## 1. 레포지토리 구조

```
cv-onboarding/
├── README.md                  # 저장소 소개 + 목차 (제일 중요)
├── .gitignore
├── environment.yml            # conda 환경 기록
├── docs/
│   ├── 01-ubuntu-install.md   # 우분투 설치 기록
│   ├── 02-nvidia-cuda.md      # 드라이버·CUDA 설치 기록
│   ├── 03-vscode-setup.md     # VS Code 설정 기록
│   └── 04-yolo-demo.md        # YOLO 실습 기록
├── img/                    # 문서에 넣을 캡처 이미지
└── src/
    ├── check_env.py
    └── yolo/
        ├── football.mp4
        ├── test.png
        ├── predict_image.py
        ├── predict_video.py
        ├── webcam_demo.py
        └── runs/
            └── detect       # 모델에서 생성된 영상 및 이미지
		        	
```

---
---


