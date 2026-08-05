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