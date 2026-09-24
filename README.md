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
│   ├── processed/    # 정제·라벨링된 데이터, recommendation/ 추천 실험 결과, evaluation/ 판정 시트·평가 결과 (Git 제외)
│   └── embeddings/   # 생성된 임베딩 결과물, 결과별 vectors.npy + manifest.json (Git 제외)
├── notebooks/        # 실험용 Jupyter Notebook (번호 순서대로 진행)
├── src/
│   ├── preprocessing/   # 음식 데이터 정제 (food_data), 사용자 조건 추출 규칙 파서 (user_query)
│   ├── labeling/        # LLM 기반 음식 속성 라벨링
│   ├── embedding/       # 문장·음식 임베딩, 결과 저장·재사용 판정
│   ├── retrieval/       # 저장된 임베딩 호환 확인, 코사인 유사도 후보 검색
│   ├── ranking/         # 필수 조건 필터, 선호 점수 재랭킹, 중복·다양성 제어
│   └── recommendation/  # 전체 파이프라인 조합, Top-K 추천, 비교 지표, 정답 세트 평가 (evaluation)
├── api/
│   └── main.py       # 추천 API 진입점
├── tests/            # 단위 테스트
├── requirements.txt
├── .gitignore
└── README.md
```

개발 흐름: Notebook에서 실험 → 검증된 기능을 `src/` 모듈로 분리 → `tests/`로 검증 → `api/`에서 사용

## 추천 파이프라인 사용 (5단계)

저장된 e5-base 임베딩(`data/embeddings/`)을 manifest 해시로 확인해 로드하고, 사용자 문장만 같은 모델로 임베딩한다.

```python
from src.embedding import E5Embedder
from src.retrieval import load_index
from src.recommendation import Recommender

index, ref = load_index("B")                       # 모델·리비전·원본 해시가 맞는 결과만 로드, 프랜차이즈 항목 제외
embedder = E5Embedder()
rec = Recommender(index, lambda t: embedder.encode_queries([t])[0], ref)
result = rec.recommend("맵지 않고 따뜻한 음식")    # 상태, 조건, 검색범위, 추천 목록(점수·근거) 포함
```

- 추천 후보는 업체명이 없는 공공 데이터 항목(1,123건)이다. 프랜차이즈 메뉴는 추천 대상에서 제외한다. `load_index(..., include_franchise=True)`로 되돌릴 수 있다.
- 조건 추출은 규칙 기반이며 지원 범위는 `src/preprocessing/user_query.py`의 `RULES` 표(`support_table()`)가 전부다.
  부정·제외 표현은 필수 조건, 긍정 표현은 선호 조건이며 '미확인' 라벨은 필수 조건을 충족하지 않는다.
- 필수 조건 통과 후보가 부족하면 전체까지, 선호 일치 항목이 부족하면 400개까지 검색 범위를 넓힌다. 필수 조건은 완화하지 않는다.
- 최종점수 = `similarity_weight × 유사도 + preference_weight × 선호점수 + menu_match_weight × 메뉴일치`. 가중치, 메뉴군 상한, 감점은 `RankingConfig`로 바꾼다.
- "치킨", "면"처럼 메뉴 종류를 말하면 그 메뉴군은 상한을 면제하고 가점(기본 0.15)을 받는다. "피자 말고"처럼 제외한 메뉴는 걸러낸다.
- 음식 라벨은 모델 추정이므로 조건 준수 지표는 저장된 라벨 기준이며 실제 정확도가 아니다.

## 평가 (6단계)

`data/processed/evaluation/judgments.csv`가 정답 세트다. 여러 설정의 상위 결과를 합친 판정 풀에 (질의, 메뉴)별 적합도(2 적합, 1 부분, 0 부적합)를 매긴다.
현재 판정은 Claude가 메뉴명·업체명·분류만 보고 매긴 모델 추정(`검토대기`)이며, 사람이 적합도를 고치고 `검토상태`를 `승인`으로 바꾸면
`06_evaluation.ipynb`가 승인 판정만으로 P@5, nDCG@5, MRR을 다시 계산한다. 풀에 없는 항목은 미판정으로 0 처리하고 미판정 비율을 함께 보고한다.
설정 간 차이는 질의 단위 부트스트랩 95% 신뢰구간으로 우연 범위인지 확인한다. 평가 질의는 `data/processed/evaluation/queries.csv`에 있다 (5단계 18개 + 추가 22개).

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
| `05_recommendation_test.ipynb` | 조건 추출·후보 검색·필터·재랭킹·중복 제어 비교 실험, 결과는 `data/processed/recommendation/` |
| `06_evaluation.ipynb`          | 판정 풀·판정 시트 관리, 설정별 P@5·nDCG@5·MRR 비교, 결과는 `data/processed/evaluation/` |
| `07_try_recommendation.ipynb`  | 사용용. 준비 셀 실행 후 `show("문장")`으로 추천 결과와 근거 확인, 맨 위 설정에서 개수·텍스트 구성·프랜차이즈 포함 변경 |
