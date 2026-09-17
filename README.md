# menu-recommendation

자연어 입력을 기반으로 음식 메뉴를 추천하는 AI 프로젝트입니다.

예: "오늘 비도 오고 쌀쌀한데 얼큰하고 따뜻한 국물 먹고 싶어" -> 음식 데이터를 바탕으로 적절한 메뉴 Top-K 추천

## 추천 파이프라인

```
자연어 입력
→ 자연어 전처리 및 특징 추출   (src/preprocessing)
→ 문장 임베딩                 (src/embedding)
→ 음식 데이터 임베딩           (src/embedding)
→ 유사도 기반 후보 검색         (src/retrieval)
→ 음식 속성을 활용한 랭킹       (src/ranking)
→ Top-K 메뉴 추천             (src/recommendation)
```

## 디렉터리 구조

```
menu-recommendation/
├── data/
│   ├── raw/          # 원본 공공 음식 데이터 (Git 제외)
│   ├── processed/    # 정제·라벨링된 데이터 (Git 제외)
│   └── embeddings/   # 생성된 임베딩 결과물 (Git 제외)
├── notebooks/        # 실험용 Jupyter Notebook (번호 순서대로 진행)
├── src/
│   ├── preprocessing/   # 자연어 전처리, 음식 데이터 정제
│   ├── embedding/       # 문장·음식 임베딩
│   ├── retrieval/       # 유사도 기반 후보 검색
│   ├── ranking/         # 음식 속성 기반 랭킹
│   └── recommendation/  # 전체 파이프라인 조합, Top-K 추천
├── api/
│   └── main.py       # 추천 API 진입점
├── tests/            # 단위 테스트
├── requirements.txt
├── .gitignore
└── README.md
```

개발 흐름: Notebook에서 실험 → 검증된 기능을 `src/` 모듈로 분리 → `tests/`로 검증 → `api/`에서 사용

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
jupyter notebook notebooks/
```

| Notebook                       | 목적                                                    |
| ------------------------------ | ------------------------------------------------------- |
| `01_data_analysis.ipynb`       | 공공 음식 데이터의 컬럼, 결측치, 분포 탐색              |
| `02_preprocessing.ipynb`       | 음식 데이터 정제·전처리 실험                            |
| `03_llm_labeling.ipynb`        | 매운맛, 국물 여부, 온도, 기름짐 등 속성 LLM 라벨링 실험 |
| `04_embedding.ipynb`           | 음식 설명·사용자 자연어 임베딩 방법 및 모델 실험        |
| `05_recommendation_test.ipynb` | 자연어 입력에 대한 후보 검색 및 추천 결과 확인          |
| `06_evaluation.ipynb`          | 추천 알고리즘 성능 비교·평가                            |
