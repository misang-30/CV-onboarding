## 0. 용어 정리
- 드라이버와 CUDA 툴킷을 둘 다 설치 해야한다.

| 구성 요소                 | 역할                                       | 없으면                      |
| --------------------- | ---------------------------------------- | ------------------------ |
| **드라이버(Driver)**      | OS가 GPU와 대화하게 해주는 커널 모듈                  | GPU를 아예 못 씀              |
| **CUDA 툴킷(Toolkit)**  | GPU용 프로그램을 **컴파일**하기 위한 도구 모음 (`nvcc` 등) | 직접 CUDA 코드를 짜거나 빌드할 수 없음 |
| **PyTorch의 CUDA 런타임** | pip으로 설치되는 PyTorch에 **이미 포함**되어 있음       | —                        |
**중요**: PyTorch를 pip으로 설치하면 필요한 CUDA 런타임이 함께 딸려옵니다. 즉 **PyTorch만 쓸 거라면 드라이버만 있으면 되고 CUDA 툴킷은 필수가 아닙니다.** 그래도 우리가 CUDA 툴킷을 설치하는 이유는, 연구를 하다 보면 커스텀 CUDA 연산자를 빌드해야 하는 논문 코드를 반드시 만나게 되고(3D 비전 쪽은 특히 흔합니다) 그때 `nvcc`가 필요하기 때문입니다.

## 2. NVIDIA 드라이버 설치
### 1). 설치
```bash
# 1. 내 GPU에 권장되는 드라이버 확인
ubuntu-drivers devices

# 2. 권장 버전 자동 설치
sudo ubuntu-drivers install

# 3. 재부팅 (필수)
sudo reboot
```

### 2). 검증

```bash
nvidia-smi
```

GPU 이름, 드라이버 버전, 우측 상단의 **CUDA Version**(이 드라이버가 지원하는 최대 CUDA 버전)이 표 형태로 나오면 성공입니다.



## 3. CUDA 툴킷

### 1). 설치

NVIDIA 공식 저장소를 사용합니다. **버전 번호는 반드시 공식 페이지에서 최신 명령을 확인하세요** (아래는 형태 예시입니다).
<https://developer.nvidia.com/cuda-downloads> → Linux → x86_64 → Ubuntu → 24.04 → deb (network)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 'cuda' 가 아니라 'cuda-toolkit-XX-X' 를 설치할 것
sudo apt install -y cuda-toolkit-12-8
```

### 2). 환경 변수 등록

설치만 하면 `nvcc` 명령을 찾지 못합니다. 경로를 셸에 알려줘야 합니다.

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version
```

> nvidia-smi는 드라이버가 지원 가능한 최대 CUDA 버전을 표시
> nvcc --version은 현재 시스템에 설치된 툴킷 버전


