# CV-onboarding  
- 컴퓨터비전 연구실 학부연구생 온보딩 과정 기록입니다.

## 환경
- Ubuntu 20.04 LTS / GTX 1080Ti / CUDA 12.4 / PyTorch 2.4+

## 목차
1. [우분투 설치](docs/01-ubuntu-install.md)
2. [NVIDIA 드라이버 · CUDA 설치](docs/02-nvidia-cuda.md)
3. [VS Code 개발 환경 설정](docs/03-vscode-setup.md)
4. [YOLO 데모 및 실험](docs/04-yolo-demo.md)

## 실행 방법
```bash
conda env create -f environment.yml
conda activate cv
python src/yolo/webcam_demo.py
