# 뉴스 기사 레이블 복구 프로젝트

> 월간 데이콘 쇼츠 - 뉴스 기사 레이블 복구 해커톤  
> 정답 레이블이 유실된 60,000건의 뉴스 기사를 Sentence-BERT로 임베딩하고, KMeans 군집화를 통해 6개 뉴스 카테고리의 레이블을 복구하는 비지도학습 프로젝트입니다.  
> 다양한 문장 임베딩 모델과 텍스트 전처리, 스케일링 및 군집 설정을 비교하여 Public Macro F1 Score를 0.467667에서 0.797515까지 개선했습니다.

***

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Competition Overview](#2-competition-overview)
- [3. Dataset](#3-dataset)
- [4. Motivation](#4-motivation)
- [5. Data Characteristics](#5-data-characteristics)
- [6. Project Pipeline](#6-project-pipeline)
- [7. Text Preprocessing](#7-text-preprocessing)
- [8. Sentence Embedding](#8-sentence-embedding)
- [9. KMeans Clustering](#9-kmeans-clustering)
- [10. Cluster Interpretation](#10-cluster-interpretation)
- [11. Experiment Strategy](#11-experiment-strategy)
- [12. Results](#12-results)
- [13. My Contribution](#13-my-contribution)
- [14. Lessons Learned](#14-lessons-learned)
- [15. Tech Stack](#15-tech-stack)
- [16. Summary](#16-summary)

***

- [1. Project Overview](#1-project-overview)
- [2. Competition Overview](#2-competition-overview)
- [3. Dataset](#3-dataset)
- [4. Motivation](#4-motivation)
- [5. Project Pipeline](#5-project-pipeline)
- [6. Feature Engineering](#6-feature-engineering)
- [7. Modeling](#7-modeling)
- [8. Results](#8-results)
- [9. My Contribution](#9-my-contribution)
- [10. Lessons Learned](#10-lessons-learned)
- [11. Tech Stack](#11-tech-stack)
- [12. Summary](#12-summary)

# 1. Project Overview

본 프로젝트는 월간 데이콘 쇼츠 - 뉴스 기사 레이블 복구 해커톤을 대상으로 수행한 뉴스 기사 비지도 분류 프로젝트입니다.

대회의 목표는 정답 레이블이 모두 유실된 뉴스 기사 데이터를 분석하여 각 기사가 속한 카테고리를 복구하는 것이었습니다.

복구해야 하는 카테고리는 다음과 같습니다.

```text
Politics
Entertainment
Business
Sports
Tech
World
```

본 프로젝트에서 가장 중요하게 본 문제는 다음과 같습니다.

> 학습용 정답 데이터가 제공되지 않기 때문에 일반적인 지도학습 분류 모델을 사용할 수 없으며,  
> 뉴스 기사의 의미를 적절하게 표현한 뒤 유사한 기사끼리 군집화해야 한다.

따라서 본 프로젝트에서는 단순히 KMeans를 적용하는 것이 아니라,

- 제목과 본문을 결합한 입력 텍스트 구성
- 뉴스 기사 텍스트 전처리
- Sentence-Transformer 모델 비교
- `all-mpnet-base-v2` 기반 문장 임베딩
- KMeans 기반 6개 군집 생성
- 군집별 대표 기사 확인
- 군집과 실제 뉴스 카테고리 매핑
- 스케일링 및 반복 군집 실험

을 통해 최종 성능을 개선하고자 했습니다.

최종적으로 Best Public Score 0.797515, Private Score 0.793410을 기록하며 **최종 8위**로 대회를 마무리했습니다.

***

# 2. Competition Overview

| 항목 | 내용 |
|---|---|
| Competition | 월간 데이콘 쇼츠 - 뉴스 기사 레이블 복구 해커톤 |
| Organizer | DACON |
| Period | 2023.09.22 ~ 2023.09.25 |
| Task | Unsupervised Text Classification |
| Objective | 레이블이 유실된 뉴스 기사의 카테고리 복구 |
| Number of Categories | 6 |
| Evaluation Metric | Macro F1 Score |
| Final Rank | 8위 |
| Private Score | 0.793410 |
| Best Public Score | 0.797515 |
| Baseline Score | 0.467667 |
| Improvement | +0.329848, 약 70.5% 향상 |
| Project Type | Individual Project |

## 2.1 Task Description

총 60,000건의 뉴스 기사에서 기존 카테고리 레이블이 모두 유실된 상황을 가정한 문제입니다.

각 뉴스 기사의 제목과 본문을 분석하여 다음 6개 카테고리 중 하나를 복구해야 합니다.

| Label | Category |
|---:|---|
| 0 | Politics |
| 1 | Entertainment |
| 2 | Business |
| 3 | Sports |
| 4 | Tech |
| 5 | World |

전체적인 문제 구조는 다음과 같습니다.

```text
News Title + Contents
          │
          ▼
Sentence Embedding
          │
          ▼
Unsupervised Clustering
          │
          ▼
Cluster Interpretation
          │
          ▼
Category Label Recovery
```

## 2.2 Evaluation Metric

본 대회의 평가 지표는 Macro F1 Score입니다.

```text
Public Score  : 전체 평가 데이터 중 30%
Private Score : 전체 평가 데이터 중 70%
```

Macro F1 Score는 각 클래스의 F1 Score를 동일한 비중으로 평균합니다.

따라서 특정 카테고리의 기사 수가 많더라도 모든 카테고리의 분류 성능을 균형 있게 높이는 것이 중요했습니다.

```text
F1 Score = Precision과 Recall의 조화평균

Macro F1 =
각 카테고리의 F1 Score를 동일한 비중으로 평균한 값
```

***

# 3. Dataset

본 프로젝트에서는 대회에서 제공된 `news.csv` 데이터를 사용했습니다.

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| `news.csv` | 60,000 | 3 | 뉴스 기사 데이터, 정답 레이블 미제공 |

데이터는 다음 컬럼으로 구성되어 있습니다.

| Column | Description |
|---|---|
| `id` | 뉴스 기사 고유 식별자 |
| `title` | 뉴스 기사 제목 |
| `contents` | 뉴스 기사 본문 |

## 3.1 Input Data

원본 데이터의 형태는 다음과 같습니다.

```text
id          : NEWS_00000
title       : Spanish coach facing action in race row
contents    : MADRID (AFP) - Spanish national team coach ...
```

별도의 학습 데이터나 정답 카테고리가 제공되지 않았기 때문에, 기사 자체에 포함된 의미적 정보만으로 카테고리를 추론해야 했습니다.

## 3.2 Text Construction

제목은 기사의 핵심 주제를 짧게 보여주고, 본문은 기사에 대한 구체적인 맥락을 제공합니다.

두 정보를 모두 활용하기 위해 제목과 본문을 하나의 텍스트로 결합했습니다.

```python
df["text"] = df["title"] + " : " + df["contents"]
```

전체 입력 구조는 다음과 같습니다.

```text
Title
  +
Contents
  │
  ▼
Combined News Text
```

제목만 사용할 경우 기사에 대한 정보가 부족할 수 있고, 본문만 사용할 경우 핵심 주제가 희석될 수 있다고 판단했습니다.

***

# 4. Motivation

## 4.1 정답 데이터가 없는 분류 문제

일반적인 텍스트 분류 문제에서는 다음과 같은 학습 데이터를 사용합니다.

```text
News Text + Category Label
             │
             ▼
Supervised Classification Model
```

그러나 본 대회에서는 정답 레이블이 제공되지 않았습니다.

```text
News Text
   │
   ▼
No Training Labels
```

따라서 지도학습 분류 모델을 직접 학습하는 대신, 뉴스 기사 간 의미적 유사성을 이용하는 비지도학습 방식이 필요했습니다.

## 4.2 핵심 문제의식

본 프로젝트는 다음 질문에서 출발했습니다.

> 뉴스 기사의 의미를 적절한 벡터로 표현할 수 있다면,  
> 같은 카테고리의 기사들이 임베딩 공간에서도 서로 가까이 모이지 않을까?

TF-IDF와 같은 단어 빈도 기반 표현은 서로 다른 단어로 작성된 유사한 기사의 관계를 충분히 반영하지 못할 수 있습니다.

예를 들어 다음 두 문장은 사용하는 단어가 다르지만 모두 기술 기사에 해당할 가능성이 높습니다.

```text
A new artificial intelligence system was released.

The company introduced its latest machine learning platform.
```

따라서 문장의 문맥과 의미를 벡터로 표현할 수 있는 Sentence-BERT 계열 모델을 활용했습니다.

***

# 5. Data Characteristics

## 5.1 Unlabeled Dataset

가장 큰 특징은 전체 60,000건의 기사에 정답 카테고리가 존재하지 않는다는 점입니다.

```text
Total News Articles : 60,000
Available Labels    : 0
Target Categories   : 6
```

모델 학습에 사용할 정답이 없기 때문에 다음 과정을 직접 수행해야 했습니다.

- 기사 의미 표현
- 유사 기사 군집화
- 군집별 대표 기사 확인
- 군집의 실제 의미 해석
- 군집 번호와 카테고리 번호 연결

## 5.2 Long Text Data

각 데이터에는 짧은 제목뿐 아니라 상대적으로 긴 기사 본문도 포함되어 있었습니다.

```text
News Article
 ├── Short Title
 └── Long Contents
```

전체 본문을 사용하면 많은 정보를 활용할 수 있지만, 기사 주제와 직접 관련이 없는 표현도 함께 포함될 수 있습니다.

따라서 제목과 본문을 함께 사용하면서 불필요한 문자열을 정리하는 과정이 필요했습니다.

## 5.3 Semantic Overlap Between Categories

일부 뉴스 카테고리는 의미적으로 명확하게 구분되지만, 서로 비슷한 표현을 공유하는 경우도 존재합니다.

예를 들어 다음과 같은 기사들은 하나의 카테고리로 명확하게 분리되지 않을 수 있습니다.

```text
기업의 기술 투자      → Business / Tech
정부의 산업 정책      → Politics / Business
스포츠 선수의 사생활  → Sports / Entertainment
국제 기업 분쟁        → World / Business
```

이러한 기사들은 군집 경계에 위치하며 전체 Macro F1 Score를 낮추는 원인이 될 수 있다고 판단했습니다.

***

# 6. Project Pipeline

전체 프로젝트 흐름은 다음과 같습니다.

```text
Raw News Data
      │
      ├── id
      ├── title
      └── contents
      │
      ▼
Title + Contents Combination
      │
      ▼
Text Preprocessing
      │
      ├── URL Removal
      ├── Hashtag Removal
      ├── Mention Removal
      ├── Number Removal
      ├── Non-ASCII Character Removal
      └── Whitespace Normalization
      │
      ▼
Sentence Embedding Model Comparison
      │
      ├── paraphrase-distilroberta-base-v1
      ├── paraphrase-MiniLM-L12-v2
      ├── paraphrase-MiniLM-L6-v2
      ├── all-roberta-large-v1
      └── all-mpnet-base-v2
      │
      ▼
Embedding Scaling Experiments
      │
      ├── StandardScaler
      ├── MinMaxScaler
      └── MaxAbsScaler
      │
      ▼
KMeans Clustering
      │
      └── n_clusters = 6
      │
      ▼
Cluster Interpretation
      │
      ├── Representative News Review
      └── Cluster-to-Category Mapping
      │
      ▼
Submission
```

***

# 7. Text Preprocessing

## 7.1 Preprocessing Rules

뉴스 기사에는 기사 주제와 직접적인 관련이 적은 문자열이 포함되어 있었습니다.

초기 전처리 함수에서는 다음 요소를 정리했습니다.

- URL
- 해시태그
- 멘션
- 숫자
- 비ASCII 문자
- 중복 공백
- 영문 대소문자

```python
import re

def preprocess_text(text):
    # URL 제거
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text,
        flags=re.MULTILINE
    )

    # 해시태그 제거
    text = re.sub(r"#\w+", "", text)

    # 멘션 제거
    text = re.sub(r"@\w+", "", text)

    # 비ASCII 문자 제거
    text = text.encode(
        "ascii",
        "ignore"
    ).decode("ascii")

    # 중복 공백 제거
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # 숫자 제거
    text = re.sub(r"\d+", "", text)

    return text.lower()
```

전처리 결과는 별도의 컬럼으로 저장했습니다.

```python
df["processed_text"] = df["text"].apply(
    preprocess_text
)
```

## 7.2 Preprocessing Experiments

초기 Baseline에서는 전처리 컬럼을 생성한 뒤 원문 텍스트를 임베딩에 사용했습니다.

후속 실험에서는 전처리 규칙을 여러 그룹으로 나누어 적용하면서 다음을 비교했습니다.

```text
Original Text
      │
      ├── 일부 전처리 규칙 적용
      ├── 전체 전처리 규칙 적용
      └── 전처리 없이 원문 사용
```

모든 전처리가 항상 성능을 높이는 것은 아니었습니다.

Sentence-BERT는 문장의 문맥을 활용하기 때문에, 과도한 전처리는 문장 구조와 의미 정보를 손상시킬 수 있습니다.

따라서 전처리 규칙을 일괄적으로 적용하기보다 제출 점수와 군집 결과를 함께 확인하며 적용 범위를 조정했습니다.

***

# 8. Sentence Embedding

## 8.1 Why Sentence-BERT?

KMeans는 숫자로 표현된 벡터를 입력받기 때문에 뉴스 기사 텍스트를 벡터로 변환해야 합니다.

Sentence-BERT는 문장 전체의 의미를 고정된 크기의 벡터로 변환합니다.

```text
News Article
     │
     ▼
Sentence-BERT
     │
     ▼
Dense Semantic Vector
```

의미가 유사한 문장일수록 임베딩 공간에서 가까운 위치에 배치될 가능성이 높습니다.

## 8.2 Embedding Model Comparison

여러 Sentence-Transformer 모델을 비교했습니다.

| Model | Public Score |
|---|---:|
| Baseline | 0.467667 |
| `paraphrase-MiniLM-L12-v2` | 0.616425 |
| `all-roberta-large-v1` | 0.696113 |
| `paraphrase-MiniLM-L6-v2` | 0.737111 |
| `all-mpnet-base-v2` | **0.782755** |

초기 Baseline에서는 다음 모델을 사용했습니다.

```python
model = SentenceTransformer(
    "paraphrase-distilroberta-base-v1"
)
```

여러 모델을 비교한 결과 `all-mpnet-base-v2`가 가장 높은 성능을 기록했습니다.

```python
model = SentenceTransformer(
    "all-mpnet-base-v2"
)

sentence_embeddings = model.encode(
    df["text"].tolist()
)
```

모델 비교 결과를 통해 군집화 알고리즘을 복잡하게 변경하는 것보다, **기사의 의미를 적절하게 표현하는 임베딩 모델을 선택하는 것이 더 중요하다**고 판단했습니다.

## 8.3 Embedding Scaling

문장 임베딩에 서로 다른 스케일링을 적용했을 때의 성능도 비교했습니다.

| Scaling Method | Public Score |
|---|---:|
| Original Embedding | 0.782755 |
| StandardScaler | 0.786907 |
| MaxAbsScaler | 0.787730 |
| MinMaxScaler | **0.787992** |

스케일링을 통해 일부 성능 향상을 얻었지만, 임베딩 모델을 변경했을 때만큼 큰 차이는 나타나지 않았습니다.

```text
Embedding Model Selection
          │
          └── Large Performance Improvement

Scaling Method
          │
          └── Small Additional Improvement
```

***

# 9. KMeans Clustering

## 9.1 Number of Clusters

복구해야 하는 뉴스 카테고리가 총 6개이므로 KMeans의 군집 수를 6으로 설정했습니다.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=6,
    random_state=SEED
)

df["kmeans_cluster"] = kmeans.fit_predict(
    sentence_embeddings
)
```

전체 군집화 구조는 다음과 같습니다.

```text
60,000 Sentence Embeddings
            │
            ▼
       KMeans, K=6
            │
            ├── Cluster 0
            ├── Cluster 1
            ├── Cluster 2
            ├── Cluster 3
            ├── Cluster 4
            └── Cluster 5
```

## 9.2 KMeans Characteristics

KMeans의 군집 번호는 실행 결과를 구분하기 위한 번호일 뿐, 실제 뉴스 카테고리 번호와 직접적인 의미 관계가 없습니다.

```text
KMeans Cluster 0 ≠ Category Label 0
```

초기 중심점과 임베딩 구조가 달라지면 동일한 카테고리가 다른 군집 번호로 생성될 수 있습니다.

따라서 KMeans 결과를 제출 레이블로 사용하기 전에 각 군집의 의미를 해석해야 했습니다.

***

# 10. Cluster Interpretation

## 10.1 Representative Article Review

군집별 기사 제목과 본문을 직접 확인하여 각 군집이 어떤 주제를 나타내는지 분석했습니다.

```python
for cluster_id in range(6):
    print(f"Cluster {cluster_id}")

    samples = df[
        df["kmeans_cluster"] == cluster_id
    ]["text"].head(10)

    for sample in samples:
        print(sample)
```

예를 들어 특정 군집에서 다음과 같은 기사가 반복적으로 나타난다면 Sports 군집으로 판단할 수 있습니다.

```text
Spanish coach facing action in race row
College Basketball: Georgia Tech, UConn Win
Game Day Preview
```

기술 관련 기사들이 집중된 군집에서는 다음과 같은 제목을 확인할 수 있었습니다.

```text
Macromedia contributes to eBay Stores
Qualcomm plans to phone it in on cellular repairs
Thomson to Back Both Blu-ray and HD-DVD
```

## 10.2 Cluster-to-Category Mapping

각 군집의 대표 기사를 확인한 뒤 실제 뉴스 카테고리 번호와 연결했습니다.

```text
KMeans Cluster
      │
      ▼
Representative Article Review
      │
      ▼
Semantic Category Interpretation
      │
      ▼
Competition Category Label
```

군집 번호는 실험마다 달라질 수 있기 때문에 고정된 매핑을 모든 실험에 동일하게 적용하지 않았습니다.

각 실험에서 다음 항목을 다시 확인했습니다.

- 군집별 대표 기사
- 군집별 기사 개수
- 카테고리가 혼합된 군집
- 군집과 실제 카테고리의 대응 관계
- Public Score 변화

이 과정은 단순히 군집 번호를 변경하는 작업이 아니라, 비지도학습 결과에 실제 문제의 의미를 부여하는 작업이었습니다.

***

# 11. Experiment Strategy

## 11.1 Model-first Experiment

대회 기간이 약 3박 4일로 짧았기 때문에 성능에 미치는 영향이 클 것으로 예상되는 요소부터 검증했습니다.

```text
Sentence Embedding Model
          │
          ▼
Text Preprocessing
          │
          ▼
Embedding Scaling
          │
          ▼
KMeans Configuration
          │
          ▼
Cluster Interpretation
```

가장 먼저 문장 임베딩 모델을 비교했고, `all-mpnet-base-v2`를 선택한 뒤 세부 실험을 진행했습니다.

## 11.2 Iterative Submissions

전체 50회의 제출을 통해 각 실험의 Public Score를 비교했습니다.

주요 성능 변화는 다음과 같습니다.

| Stage | Submission | Public Score |
|---|---|---:|
| Baseline | `baseline_submit.csv` | 0.467667 |
| MiniLM-L12 | `paraphrase-MiniLM-L12-v2.csv` | 0.616425 |
| RoBERTa | `all-roberta-large-v1.csv` | 0.696113 |
| MiniLM-L6 | `paraphrase-MiniLM-L6-v2.csv` | 0.737111 |
| MPNet | `all-mpnet-base-v2.csv` | 0.782755 |
| StandardScaler | `all-mpnet-base-v2standard.csv` | 0.786907 |
| MaxAbsScaler | `all-mpnet-base-v2MaxAbs.csv` | 0.787730 |
| MinMaxScaler | `all-mpnet-base-v2MinMax.csv` | 0.787992 |
| Iteration | `all-mpnet-base-v2_29.csv` | 0.797111 |
| Iteration | `all-mpnet-base-v2_32.csv` | 0.797221 |
| Best Public | `all-mpnet-base-v2_36.csv` | **0.797515** |

성능 변화는 다음과 같습니다.

```text
Baseline
0.467667
    │
    ├── Sentence Embedding Model Comparison
    │       0.616425
    │       0.696113
    │       0.737111
    │
    ├── all-mpnet-base-v2
    │       0.782755
    │
    ├── Scaling Experiments
    │       0.786907
    │       0.787730
    │       0.787992
    │
    └── Repeated Clustering Experiments
            0.797515
```

## 11.3 Experiment Selection

Public Score만을 기준으로 실험 결과를 판단하지 않았습니다.

다음 요소를 함께 고려했습니다.

- Public Score
- 군집별 주제 일관성
- 특정 카테고리의 혼합 정도
- 군집 크기의 과도한 불균형 여부
- Public과 Private 성능의 안정성

짧은 대회에서는 많은 모델을 무작정 실행하기보다, 하나의 요소를 변경하고 점수 변화를 확인하는 방식으로 실험을 진행했습니다.

***

# 12. Results

## 12.1 Final Performance

| Metric | Result |
|---|---:|
| Baseline Public Score | 0.467667 |
| Best Public Score | **0.797515** |
| Private Score | **0.793410** |
| Final Rank | **8위** |
| Number of Submissions | **50회** |

Baseline 대비 Best Public Score는 다음과 같이 개선되었습니다.

```text
Baseline Public Score : 0.467667
Best Public Score     : 0.797515
Absolute Improvement : 0.329848
Relative Improvement : 약 70.5%
```

Public과 Private 점수 차이는 다음과 같습니다.

```text
Public Score  : 0.797515
Private Score : 0.793410
Score Gap     : 0.004105
```

Public과 Private 점수 차이가 크지 않아 최종 평가 데이터에서도 비교적 안정적인 성능을 유지했습니다.

## 12.2 Key Performance Factors

성능 개선에 가장 크게 기여한 요소는 다음과 같습니다.

```text
1. Sentence-BERT 모델 비교
2. all-mpnet-base-v2 선택
3. 제목과 본문을 결합한 입력 구성
4. 전처리 규칙별 비교
5. 임베딩 스케일링 비교
6. 군집별 대표 기사 분석
7. 반복적인 군집-카테고리 매핑 검증
```

특히 임베딩 모델을 변경했을 때 가장 큰 성능 향상이 나타났습니다.

```text
Baseline                  : 0.467667
all-mpnet-base-v2         : 0.782755
Additional Optimization   : 0.797515
```

***

# 13. My Contribution

본 프로젝트의 데이터 분석, 전처리, 모델링 및 군집 해석 과정을 개인으로 수행했습니다.

## 13.1 Data Analysis

- 뉴스 기사 데이터 구조 분석
- `title`과 `contents` 컬럼의 역할 확인
- 제목과 본문 결합 방식 설계
- 기사 텍스트 내부의 불필요한 문자열 확인
- 군집별 기사 분포 및 대표 기사 분석

## 13.2 Text Processing

- URL 제거
- 해시태그 및 멘션 제거
- 숫자 제거
- 비ASCII 문자 제거
- 중복 공백 정리
- 전처리 규칙 조합별 비교

## 13.3 Modeling

- Sentence-Transformer 계열 모델 비교
- `all-mpnet-base-v2` 기반 문장 임베딩 생성
- KMeans 기반 6개 뉴스 군집 생성
- StandardScaler 적용
- MinMaxScaler 적용
- MaxAbsScaler 적용
- 반복적인 군집 설정 실험

## 13.4 Cluster Analysis

- 군집별 대표 기사 직접 확인
- 군집의 의미적 주제 해석
- 군집과 실제 뉴스 카테고리 매핑
- 카테고리가 혼합된 군집 분석
- 실험별 군집 번호 변화 검증

## 13.5 Experiment Management

- 총 50회의 제출 결과 비교
- 임베딩 모델별 성능 기록
- 전처리 및 스케일링 실험 관리
- Public Score 기반 성능 변화 추적
- 최종 제출 결과 선정

***

# 14. Lessons Learned

## 14.1 비지도 분류에서는 표현 방법이 중요하다

동일한 KMeans를 사용하더라도 임베딩 모델에 따라 성능이 크게 달라졌습니다.

```text
Baseline                 : 0.467667
paraphrase-MiniLM-L6-v2 : 0.737111
all-mpnet-base-v2       : 0.782755
```

군집화 알고리즘을 복잡하게 변경하기 전에 데이터의 의미를 잘 보존하는 벡터 표현을 선택하는 것이 중요하다는 점을 확인했습니다.

***

## 14.2 전처리는 항상 많을수록 좋은 것이 아니다

텍스트 전처리를 많이 적용하면 노이즈를 제거할 수 있지만, 문장 구조와 의미 정보도 함께 손실될 수 있습니다.

```text
Insufficient Preprocessing
          │
          └── Noise Remains

Excessive Preprocessing
          │
          └── Semantic Information Loss
```

Sentence-BERT 기반 문장 임베딩에서는 각 전처리 규칙이 실제로 성능에 도움이 되는지 개별적으로 확인해야 한다는 점을 배웠습니다.

***

## 14.3 군집 결과에는 해석 과정이 필요하다

KMeans는 군집 번호만 생성하며, 해당 번호가 어떤 뉴스 카테고리인지는 알려주지 않습니다.

따라서 군집별 기사를 직접 확인하고 실제 문제의 레이블 체계와 연결하는 과정이 필요했습니다.

```text
Cluster Number
      │
      ▼
Human Interpretation
      │
      ▼
Category Meaning
```

이를 통해 비지도학습 결과를 실제 문제에서 활용 가능한 형태로 변환하는 과정을 경험했습니다.

***

## 14.4 하나의 실행 결과에 의존해서는 안 된다

KMeans는 초기 중심점과 입력 임베딩 구조에 따라 군집 결과가 달라질 수 있습니다.

따라서 한 번의 실행 결과만 사용하는 것이 아니라 반복적인 실험을 통해 다음을 확인했습니다.

- 군집의 의미적 일관성
- 군집별 데이터 분포
- 카테고리 간 혼합 정도
- Public Score 변동
- 최종 평가 성능의 안정성

***

## 14.5 짧은 대회에서는 실험 우선순위가 중요하다

대회 기간이 짧았기 때문에 모든 NLP 모델과 군집화 알고리즘을 시도할 수는 없었습니다.

이에 따라 성능에 미치는 영향이 클 것으로 예상되는 순서로 실험했습니다.

```text
Embedding Model
      │
      ▼
Preprocessing
      │
      ▼
Scaling
      │
      ▼
Clustering
      │
      ▼
Label Mapping
```

제한된 시간 안에서는 복잡한 모델을 무작정 추가하기보다, 실험 결과를 빠르게 확인하고 다음 가설을 세우는 과정이 중요하다는 점을 경험했습니다.

***

# 15. Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Text Processing | Regular Expression |
| NLP | Sentence-Transformers |
| Embedding | all-mpnet-base-v2 |
| Machine Learning | Scikit-learn |
| Clustering | KMeans |
| Scaling | StandardScaler, MinMaxScaler, MaxAbsScaler |
| Environment | Google Colab, Jupyter Notebook |
| Version Control | Git, GitHub |

***

# 16. Summary

본 프로젝트에서는 정답 레이블이 제공되지 않은 60,000건의 뉴스 기사를 대상으로 Sentence-BERT와 KMeans를 결합하여 뉴스 카테고리 레이블을 복구했습니다.

제목과 본문을 결합하여 기사 전체의 의미를 표현했으며, 여러 Sentence-Transformer 모델을 비교한 결과 `all-mpnet-base-v2`가 가장 높은 성능을 보였습니다.

이후 텍스트 전처리, 임베딩 스케일링, KMeans 군집화 및 군집별 기사 해석을 반복하면서 Public Macro F1 Score를 0.467667에서 0.797515까지 개선했습니다.

최종적으로 Private Score 0.793410을 기록하며 대회를 8위로 마무리했습니다.

본 프로젝트를 통해 다음 역량을 발전시킬 수 있었습니다.

- 정답이 없는 데이터에 대한 비지도학습 문제 설계 역량
- Sentence-BERT 기반 문장 임베딩 활용 역량
- 여러 사전학습 모델을 비교하고 선택하는 실험 역량
- 군집 결과의 의미를 분석하고 실제 레이블로 변환하는 역량
- 제한된 기간 안에 실험 우선순위를 설정하는 능력
- 반복적인 제출 결과를 바탕으로 성능을 개선하는 능력

> 복잡한 모델을 적용하기 전에 데이터의 의미를 적절하게 표현하고,  
> 비지도학습으로 생성된 결과를 문제의 맥락 안에서 해석하는 것이 중요하다는 점을 배웠습니다.

***

## Repository Structure

```text
뉴스 기사 레이블 복구 해커톤/
│
├── README.md
│
└── notebooks/
    ├── [Baseline]_SentenceBERT + KMeans.ipynb
    ├── paraphrase-MiniLM-L12-v2_.ipynb
    ├── paraphrase-MiniLM-L6-v2.ipynb
    ├── all-mpnet-base-v1.ipynb
    ├── all-mpnet-base-v2.ipynb
    ├── all-mpnet-base-v2standard.ipynb
    ├── all-mpnet-base-v2MinMax.ipynb
    ├── all-mpnet-base-v2MaxAbs.ipynb
    ├── all-mpnet-base-v2_preprocess12345.ipynb
    ├── all-mpnet-base-v2_preprocess6789_10.ipynb
    ├── all-mpnet-base-v2_11.ipynb
    ├── ...
    └── all-mpnet-base-v2_38.ipynb
```

## Links

- [Competition Page](https://dacon.io/competitions/official/236159/overview/description)
- [Experiment Notebooks](./notebooks)
