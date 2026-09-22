# menu-recommendation

자연어 입력을 기반으로 음식 메뉴를 추천하는 AI 프로젝트입니다.

예: "오늘 비도 오고 쌀쌀한데 얼큰하고 따뜻한 국물 먹고 싶어" -> 음식 데이터를 바탕으로 적절한 메뉴 Top-K 추천

## 추천 파이프라인

```
자연어 입력
→ 자연어 전처리 및 특징 추출   (recommender/preprocessing)
→ 문장 임베딩                 (recommender/embedding)
→ 음식 데이터 임베딩           (recommender/embedding)
→ 유사도 기반 후보 검색         (recommender/retrieval)
→ 음식 속성을 활용한 랭킹       (recommender/ranking)
→ Top-K 메뉴 추천             (recommender/recommendation)
```

## 디렉터리 구조

```text
Menu_recommend/
├── frontend/                 # 화면 구현 영역 (아직 미구현)
├── backend/
│   ├── app/main.py           # API 진입점 (현재 구조 확인용)
│   └── requirements.txt      # 백엔드 의존성
├── recommender/              # 추천 엔진 Python 패키지
│   ├── preprocessing/        # 음식 데이터·입력 전처리
│   ├── embedding/            # 임베딩
│   ├── retrieval/            # 후보 검색
│   ├── ranking/              # 후보 정렬
│   ├── recommendation/       # 추천 파이프라인
│   ├── tests/                # 추천 엔진 테스트
│   └── requirements.txt
├── experiments/
│   ├── notebooks/            # 01~06 단계별 실험
│   ├── labeling/             # 오프라인 라벨링·검증
│   ├── tests/                # 라벨링 테스트
│   └── requirements.txt
├── data/                     # 기존 데이터 경로 유지 (Git 제외)
│   ├── raw/
│   ├── processed/labeling/
│   └── embeddings/
├── tests/                    # 프로젝트 통합·import 검사
├── pytest.ini
└── requirements.txt          # 전체 Python 개발 의존성
```

프런트엔드 -> 백엔드 API -> 추천 엔진 순서로 연결합니다. 현재는 코드를 분리한 구조이며, 추천 엔진을 별도 서버로 실행하지 않습니다. 백엔드는 `recommender` 패키지를 import합니다. 추천 엔진은 백엔드·실험 코드에 의존하지 않습니다.

노트북에서 실험한 추천 로직은 `recommender/`로 옮깁니다. 데이터 준비용 라벨링은 `experiments/labeling/`에서 수행하며 온라인 추천 요청과 분리합니다.

### 기존 경로에서 변경된 위치

| 기존 | 현재 |
|---|---|
| `api/` | `backend/app/` |
| `src/` (labeling 제외) | `recommender/` |
| `src/labeling/` | `experiments/labeling/` |
| `notebooks/` | `experiments/notebooks/` |
| `tests/test_food_data.py` | `recommender/tests/test_food_data.py` |
| `tests/test_labeling.py` | `experiments/tests/test_labeling.py` |

기존 `from src...` import는 `from recommender...`로, 라벨링은 `from experiments.labeling...`으로 바꿉니다. 아래 명령은 저장소 루트에서 실행합니다.

```bash
python -m backend.app.main
python -m pytest -q
```

백엔드 진입점은 아직 HTTP 서버가 아닙니다. API 구현 시 프레임워크를 추가합니다. 노트북은 저장소 루트를 탐색해 `data/`에 접근하며 저장된 실행 출력은 이전 실험 기록으로 보존합니다.

## Dataset

이 프로젝트는 [공공데이터포털](https://www.data.go.kr)에서 제공하는 음식 관련 공공데이터(식품 영양성분 데이터)를 사용합니다.

원본 데이터, 전처리 결과, 임베딩 결과물은 용량 및 라이선스 문제로 저장소에 포함하지 않으며 `.gitignore`로 제외됩니다.
디렉터리 구조만 `.gitkeep` 파일로 유지됩니다.

### 데이터 준비

1. 공공데이터포털에서 음식 데이터 CSV를 직접 다운로드합니다.
2. 다운로드한 CSV 파일을 `data/raw/` 아래에 배치합니다.

```
data/
├── raw/
│   ├── .gitkeep
│   └── food_nutrition.csv   # 공공데이터포털에서 다운로드한 원본 CSV
├── processed/
│   └── .gitkeep
└── embeddings/
    └── .gitkeep
```

### 디렉터리 역할

| 디렉터리 | 역할 |
|---|---|
| `data/raw/` | 공공데이터포털에서 다운로드한 원본 데이터를 저장합니다. 원본 데이터는 수정하지 않습니다. |
| `data/processed/` | 전처리, 정제, LLM 라벨링 등을 거친 데이터가 생성됩니다. |
| `data/embeddings/` | 음식 데이터를 임베딩한 벡터 파일과 관련 결과물이 생성됩니다. |

## 개발 환경 설정

Python 3.11 이상이 필요합니다.

```bash
# 가상환경 생성 및 활성화
python3.12 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 테스트 실행
pytest
```

## Jupyter Notebook 실행

```bash
source .venv/bin/activate
jupyter notebook experiments/notebooks/
```

| Notebook                       | 목적                                                    |
| ------------------------------ | ------------------------------------------------------- |
| `01_data_analysis.ipynb`       | 공공 음식 데이터의 컬럼, 결측치, 분포 탐색              |
| `02_preprocessing.ipynb`       | 음식 데이터 정제·전처리 실험                            |
| `03_llm_labeling.ipynb`        | 매운맛, 국물 여부, 온도, 기름짐 등 속성 LLM 라벨링 실험 |
| `04_embedding.ipynb`           | 음식 설명·사용자 자연어 임베딩 방법 및 모델 실험        |
| `05_recommendation_test.ipynb` | 자연어 입력에 대한 후보 검색 및 추천 결과 확인          |
| `06_evaluation.ipynb`          | 추천 알고리즘 성능 비교·평가                            |
