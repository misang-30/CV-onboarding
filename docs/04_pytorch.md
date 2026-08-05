## 0.아나콘다 설치

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 라이선스 동의 → 경로 기본값 → conda init: yes
exec $SHELL
```



## 1.파이토치 전용 가상환경 만들기
``` bash
# 1. pytorch_env라는 이름의 파이썬 3.10 방 만들기
conda create -n pytorch_env python=3.10 -y

# 2. 만든 방으로 들어가기
conda activate pytorch_env
```
> **철칙 1: base 환경에는 아무것도 설치하지 않습니다.**
> **철칙 2: 프로젝트 하나에 환경 하나.** 나중에 두 프로젝트가 서로 다른 버전의 라이브러리를 요구할 때, 환경을 분리해두지 않았다면 둘 다 망가집니다. 이건 언젠가 반드시 겪습니다.



## 2.파이토치 설치하기

``` bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


## 3.설치가 잘 되었는지 체크.
```python
import torch

print("torch version :", torch.__version__)
print("built with cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu name      :", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok     :", (x @ x).shape)
else:
    print("GPU를 사용할 수 없습니다. 멘토에게 이 출력 전체를 보여주세요.")
```
`cuda available: False`가 나오면 **넘어가지 말고** 여기서 해결합니다. 이 상태로 진행하면 이후 모든 실습이 CPU로 돌아 수십 배 느려집니다.