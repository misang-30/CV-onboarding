## 1. VS Code 설치

```bash
sudo snap install code --classic
```

또는 <https://code.visualstudio.com> 에서 `.deb` 파일을 받아 설치합니다.


## 2. 필수 확장 프로그램

`Ctrl+Shift+X`로 확장 탭을 열고 설치합니다.

| 확장                      | 용도                             |
| ----------------------- | ------------------------------ |
| **Python** (Microsoft)  | 필수. 실행·디버깅·인터프리터 선택            |
| **Pylance**             | 자동완성, 타입 힌트                    |
| **Jupyter**             | `.ipynb` 노트북을 VS Code 안에서 실행   |
| **GitLens**             | 각 줄을 누가 언제 왜 바꿨는지 표시           |
| **Ruff**                | 코드 스타일 자동 정리                   |
| **Markdown All in One** | 마크다운 미리보기·목차 자동 생성 (문서 작성에 필수) |
| **Error Lens**          | 에러를 코드 옆에 바로 표시                |

## 3. (중요) 인터프리터 선택

`Ctrl+Shift+P` → `Python: Select Interpreter` → **`cv` 환경**을 선택합니다.

`ModuleNotFoundError: No module named 'torch'`가 뜨는 원인의 대부분은 코드가 잘못된 게 아니라 **VS Code가 다른 파이썬을 보고 있는 것**입니다. 좌측 하단 상태바에 현재 선택된 환경이 표시되니 항상 확인하세요.


## 4. (중요) 디버깅 : 중단 점 사용
check_env.py`를 열고, 줄 번호 왼쪽을 클릭해 **빨간 점(중단점)** 을 찍은 뒤 `F5`로 실행해보세요.

- `F10` 한 줄 실행 / `F11` 함수 안으로 들어가기 / `F5` 계속
- 좌측 **VARIABLES** 패널에서 그 시점의 모든 변수 값을 볼 수 있습니다
- 하단 **DEBUG CONSOLE**에서 `x.shape`, `x.dtype`처럼 직접 쳐볼 수 있습니다

딥러닝 코드는 텐서의 shape과 dtype이 안 맞아서 터지는 경우가 대부분인데, `print`를 넣었다 지웠다 하는 것보다 중단점을 찍고 shape을 직접 들여다보는 쪽이 압도적으로 빠릅니다. 


## 5. 개발 환경 저장 방법

'conda env export > environment.yml' 로 환경을 파일로 저장할 수 있다