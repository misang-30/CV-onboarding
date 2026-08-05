
### 1. 설치 USB 만들기
- 버전을 선택해야한다.
1. <https://releases.ubuntu.com/24.04/> 에서 Desktop 이미지(`.iso`) 다운로드
2. **Rufus** (<https://rufus.ie>) 또는 **balenaEtcher**로 USB(8GB 이상)에 굽기
3. Rufus 설정: 파티션 방식 **GPT**, 대상 시스템 **UEFI** (0-1에서 확인한 부팅 방식에 맞춤)

### 2. 설치 진행

1. 재부팅 → 부팅 시 `F2`/`F10`/`F12`/`Del` (제조사마다 다름)로 BIOS 진입
2. **Secure Boot를 Disabled로 변경** — 나중에 NVIDIA 드라이버 설치할 때 문제를 크게 줄여줍니다 (이유는 Day 2에서 설명)
3. 부팅 순서에서 USB를 최우선으로
4. "Try or Install Ubuntu" 선택
5. 설치 유형 선택
   - **듀얼 부팅**: "Install Ubuntu alongside Windows Boot Manager" 선택 후 슬라이더로 우분투에 100GB 이상 할당
   - **우분투만**: "Erase disk and install Ubuntu" (윈도우가 완전히 지워집니다)
6. 설치 중 "Install third-party software..." 항목은 **체크**합니다
7. 사용자 계정 생성 — **비밀번호를 반드시 기억하세요.** `sudo` 명령마다 물어봅니다.

### 3. 설치 직후 첫 작업

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget vim htop
```

- `build-essential`은 컴파일러 모음으로, 나중에 드라이버 빌드에 필요합니다.

## 4. 완료 확인
-  우분투 데스크탑으로 부팅된다
-  (듀얼 부팅이라면) 재부팅 시 GRUB 메뉴에서 윈도우도 선택할 수 있다
- `lspci | grep -i nvidia` 실행 시 내 그래픽카드 이름이 출력된다.
