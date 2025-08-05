## 🚀 프로젝트 실행 방법

YOLOv5 기반의 음식 이미지 분류 모델을 실행하기 위한 기본 환경 설정 및 실행 순서입니다.

---

### 1. 프로젝트 클론

먼저 GitHub에서 프로젝트를 클론합니다.

```bash
git clone https://github.com/Kang-Minseokk/Dalton-AI.git
cd Dalton-AI/yolov5-master
2. 데이터셋 다운로드 및 설정
Google Drive에서 제공되는 food_image 디렉토리를 다운로드하고, 압축을 해제한 후 다음 위치에 옮깁니다:

bash
복사
편집
yolov5-master/data/food_image/
💡 추후 자동 다운로드 스크립트가 제공될 예정입니다.

3. (선택 사항) Conda 환경 활성화
conda를 사용하는 경우, 미리 만들어둔 환경을 활성화합니다.

bash
복사
편집
conda activate your-env-name
4. 필요한 패키지 설치
requirements.txt를 기준으로 필요한 라이브러리를 설치합니다.

bash
복사
편집
pip install -r requirements.txt
5. 학습 실행
환경이 준비되었으면 학습을 시작합니다.

bash
복사
편집
python train.py
✅ 참고
data/ 디렉토리는 .gitignore 처리되어 GitHub에는 업로드되지 않습니다.

학습 결과는 runs/train/exp/ 디렉토리에 저장됩니다.

모델 학습 완료 후 best.pt 파일을 추론에 사용할 수 있습니다.

yaml
복사
편집

---

필요하시면 이후에 이어서 `추론 방법`, `모델 변환`, `CoreML 적용` 같은 섹션도 이어서 작성해 드릴 수 있어요!
