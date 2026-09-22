# Menu Recommendation

자연어 입력을 기반으로 음식 메뉴를 추천하는 프로젝트입니다.

## 폴더 구조

```text
Menu_recommend/
├── frontend/                   # 사용자 화면 (아직 미구현)
├── backend/                    # 백엔드 API
│   ├── app/main.py             # 현재 구조 확인용 진입점
│   └── requirements.txt
├── ai/                         # 추천 엔진·실험·데이터
│   ├── recommender/
│   │   ├── preprocessing/
│   │   ├── embedding/
│   │   ├── retrieval/
│   │   ├── ranking/
│   │   ├── recommendation/
│   │   └── tests/
│   ├── experiments/
│   │   ├── notebooks/          # 01 분석 ~ 06 평가
│   │   ├── labeling/           # 오프라인 음식 속성 라벨링
│   │   └── tests/
│   ├── data/
│   │   ├── raw/                # 원본 CSV
│   │   ├── processed/labeling/ # 정제·라벨·진행 기록
│   │   └── embeddings/         # 임베딩 결과
│   ├── tests/                  # 패키지 통합 검사
│   └── requirements.txt
├── requirements.txt            # 전체 개발 의존성
├── pytest.ini
└── README.md
```

프런트엔드 -> 백엔드 API -> AI 추천 엔진 순서로 연결합니다. 현재는 코드 영역을 분리한 구조이며 추천 엔진을 별도 서버로 실행하지 않습니다. 백엔드는 `ai.recommender`를 import합니다. 추천 엔진은 백엔드나 실험 모듈에 의존하지 않습니다.

데이터 준비용 라벨링은 `ai.experiments.labeling`에 있으며 온라인 추천 요청과 분리합니다. 프런트엔드 프레임워크와 실제 HTTP API는 아직 구현되지 않았습니다.

## 실행

Python 3.11 이상을 사용하며 아래 명령은 저장소 루트에서 실행합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m backend.app.main
jupyter notebook ai/experiments/notebooks/
```

`backend.app.main`은 현재 구조 확인 메시지만 출력합니다. 실제 서버 실행 명령은 API 구현 시 추가합니다. 노트북은 저장소 루트를 탐색해 `ai/data/`에 접근합니다. 이동 전 실행 출력은 실험 기록으로 보존돼 과거 경로가 표시될 수 있습니다.

## 변경된 경로

| 이전 | 현재 |
|---|---|
| `recommender/` | `ai/recommender/` |
| `experiments/` | `ai/experiments/` |
| `data/` | `ai/data/` |
| `tests/` | `ai/tests/` |

기존 `from recommender...`는 `from ai.recommender...`로, `from experiments...`는 `from ai.experiments...`로 변경합니다. `backend/`와 `frontend/`는 그대로 사용합니다.

라벨링 작업을 이어갈 때 입력·출력 경로는 `ai/data/processed/labeling/`입니다. 기존 파일 내용과 진행 상태는 유지됩니다.

## 데이터와 실험

원본 공공 음식 CSV는 `ai/data/raw/food_nutrition.csv`에 둡니다. 원본·정제·라벨링·임베딩 데이터는 Git에서 제외하며 `.gitkeep`으로 폴더만 유지합니다.

노트북은 아래 순서로 사용합니다.

1. `01_data_analysis.ipynb`: 데이터 탐색
2. `02_preprocessing.ipynb`: 정제·분류
3. `03_llm_labeling.ipynb`: 속성 라벨링
4. `04_embedding.ipynb`: 임베딩 실험
5. `05_recommendation_test.ipynb`: 후보 검색·추천
6. `06_evaluation.ipynb`: 성능 평가

검증된 추천 로직은 `ai/recommender/`에 구현하고, 영역별 테스트를 실행한 후 백엔드에 연결합니다.
