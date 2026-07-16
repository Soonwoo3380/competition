````markdown
# 뉴스 기사 레이블 복구 해커톤

> Sentence-BERT를 활용해 레이블이 유실된 뉴스 기사를 임베딩하고,  
> KMeans 군집화를 통해 6개 뉴스 카테고리의 레이블을 복구한 프로젝트입니다.

- **대회명:** 월간 데이콘 쇼츠 - 뉴스 기사 레이블 복구 해커톤
- **주최:** DACON
- **진행 기간:** 2023.09.22 ~ 2023.09.25
- **평가 지표:** Macro F1 Score
- **최종 순위:** 8위
- **Best Public Score:** 0.797515
- **Private Score:** 0.793410
- **제출 횟수:** 50회

<br>

## 1. Project Overview

본 대회의 목표는 레이블이 유실된 뉴스 기사 데이터를 분석하여 각 기사가 속한 카테고리를 복구하는 것입니다.

정답 레이블이 제공되지 않았기 때문에 일반적인 지도학습 분류 모델을 사용할 수 없었습니다. 이에 뉴스 기사의 제목과 본문을 문장 임베딩으로 변환하고, 의미적으로 유사한 기사들을 군집화하는 비지도학습 방식으로 문제에 접근했습니다.

전체적인 분석 과정은 다음과 같습니다.

```text
뉴스 기사 제목 및 본문
        ↓
데이터 탐색 및 텍스트 전처리
        ↓
Sentence-BERT 문장 임베딩
        ↓
KMeans 기반 6개 군집 생성
        ↓
군집별 대표 기사 확인
        ↓
군집과 뉴스 카테고리 매핑
        ↓
제출 및 반복 실험
```

<br>

## 2. Competition Overview

### Problem

총 60,000건의 뉴스 기사에 존재했던 레이블이 유실된 상황을 가정합니다.

각 뉴스 기사의 제목과 본문을 바탕으로 다음 6개 카테고리 중 하나를 복구해야 합니다.

| Label | Category |
|---:|---|
| 0 | Business |
| 1 | Entertainment |
| 2 | Politics |
| 3 | Sports |
| 4 | Tech |
| 5 | World |

### Evaluation

평가 지표는 각 클래스의 F1 Score를 동일한 비중으로 평균하는 **Macro F1 Score**입니다.

```text
Public Score  : 전체 평가 데이터 중 30%
Private Score : 전체 평가 데이터 중 70%
```

Macro F1은 특정 카테고리의 데이터 수가 많더라도 모든 카테고리의 분류 성능을 동일한 비중으로 반영합니다.

<br>

## 3. Dataset

뉴스 기사 데이터는 다음과 같은 정보로 구성되어 있습니다.

| Column | Description |
|---|---|
| `id` | 뉴스 기사 고유 ID |
| `title` | 뉴스 기사 제목 |
| `contents` | 뉴스 기사 본문 |

별도의 학습 데이터나 정답 레이블이 제공되지 않았기 때문에, 뉴스 기사 자체가 가진 의미적 구조를 활용해 카테고리를 추론해야 했습니다.

### Input Text Construction

제목은 기사의 핵심 주제를 압축해서 보여주고, 본문은 기사에 대한 구체적인 맥락을 제공합니다.

두 정보를 모두 활용하기 위해 제목과 본문을 하나의 텍스트로 결합했습니다.

```python
df["text"] = df["title"] + ". " + df["contents"]
```

제목만 사용할 경우 정보가 부족할 수 있고, 본문만 사용할 경우 기사의 핵심 주제가 희석될 수 있다고 판단했습니다.

<br>

## 4. Motivation

이 문제에서 가장 중요한 것은 단순히 KMeans를 적용하는 것이 아니라, **기사 간 의미적 유사성을 얼마나 잘 표현할 수 있는가**라고 판단했습니다.

TF-IDF와 같은 빈도 기반 표현은 동일한 주제를 다루면서도 서로 다른 단어를 사용하는 기사들의 관계를 충분히 반영하기 어렵습니다.

예를 들어 다음 두 문장은 사용하는 단어는 다르지만 의미적으로는 모두 기술 뉴스에 해당할 수 있습니다.

```text
A new artificial intelligence system was released.

The company introduced its latest machine learning platform.
```

따라서 문장의 문맥과 의미를 벡터로 표현할 수 있는 Sentence-BERT 계열 모델을 활용하고, 여러 사전학습 임베딩 모델의 성능을 비교했습니다.

<br>

## 5. Project Pipeline

```text
1. 데이터 구조 확인
   └─ 제목, 본문, 기사 길이 및 반복 문자열 탐색

2. 입력 텍스트 구성
   └─ title과 contents 결합

3. 텍스트 전처리
   ├─ HTML 관련 문자열 제거
   ├─ URL 및 링크 표현 제거
   ├─ 언론사와 통신사 표기 제거
   ├─ 숫자와 불필요한 특수문자 제거
   └─ 중복 공백 및 불용어 제거

4. 문장 임베딩
   ├─ paraphrase-MiniLM-L12-v2
   ├─ paraphrase-MiniLM-L6-v2
   ├─ all-roberta-large-v1
   └─ all-mpnet-base-v2

5. 군집화
   └─ KMeans, n_clusters=6

6. 군집 해석
   ├─ 군집별 기사 제목 확인
   ├─ 군집별 주요 주제 파악
   └─ 뉴스 카테고리와 군집 매핑

7. 반복 실험
   ├─ 임베딩 모델 비교
   ├─ 텍스트 전처리 변경
   ├─ 스케일링 방식 비교
   └─ 군집 결과 반복 검증
```

<br>

## 6. Approach

### 6.1 Text Preprocessing

뉴스 기사 본문에는 기사 주제와 직접적인 관련이 없는 문자열이 반복적으로 포함되어 있었습니다.

대표적으로 다음 요소들을 확인하고 제거했습니다.

- URL 및 링크 관련 문자열
- HTML 태그 및 HTML Entity
- `HREF`, `html`, `short_description` 등의 반복 문자열
- `AP`, `Reuters` 등의 통신사 표기
- 해시태그와 멘션
- 숫자 및 불필요한 특수문자
- 영어 불용어
- 중복 공백

모든 문자를 일괄적으로 제거하기보다는 실제 기사 데이터에서 반복적으로 발견되는 노이즈를 확인한 뒤 전처리 규칙을 추가했습니다.

```python
import re

def preprocess_text(text):
    text = str(text).lower()

    # HTML 태그 및 HTML Entity 제거
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # URL 및 링크 관련 문자열 제거
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"href\s*=\s*[^\s]+", " ", text)

    # 반복적으로 등장하는 문자열 제거
    text = re.sub(r"\bhtml\b", " ", text)
    text = re.sub(r"\bshort_description\b", " ", text)

    # 통신사 표기 제거
    text = re.sub(r"\(ap\)|\(reuters\)|\bap\b", " ", text)

    # 숫자, 해시태그 및 특수문자 제거
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[/\\;$#]", " ", text)

    # 중복 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

<br>

### 6.2 Sentence Embedding Model Comparison

뉴스 기사의 의미를 벡터로 표현하기 위해 여러 Sentence-Transformer 모델을 비교했습니다.

| Model | Public Score |
|---|---:|
| Baseline | 0.467667 |
| `paraphrase-MiniLM-L12-v2` | 0.616425 |
| `all-roberta-large-v1` | 0.696113 |
| `paraphrase-MiniLM-L6-v2` | 0.737111 |
| `all-mpnet-base-v2` | 0.782755 |

실험 결과 `all-mpnet-base-v2`가 다른 임베딩 모델보다 높은 성능을 보였습니다.

이를 통해 군집화 알고리즘을 복잡하게 변경하는 것보다, **기사 간 의미적 관계를 잘 표현하는 임베딩 모델을 선택하는 것이 성능에 더 큰 영향을 준다**고 판단했습니다.

최종적으로 `all-mpnet-base-v2`를 중심으로 후속 실험을 진행했습니다.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

embeddings = model.encode(
    df["processed_text"].tolist(),
    show_progress_bar=True
)
```

<br>

### 6.3 KMeans Clustering

복구해야 하는 뉴스 카테고리가 총 6개이므로 KMeans의 군집 수를 6개로 설정했습니다.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=6,
    random_state=42
)

clusters = kmeans.fit_predict(embeddings)
```

Sentence-BERT 임베딩 공간에서는 의미적으로 유사한 기사들이 가까운 위치에 존재할 가능성이 높습니다.

KMeans를 이용해 전체 뉴스 기사를 6개의 군집으로 나누고, 각 군집에 포함된 기사 제목과 본문을 확인했습니다.

<br>

### 6.4 Cluster Interpretation

KMeans가 생성하는 군집 번호는 실제 뉴스 카테고리 번호와 직접적으로 일치하지 않습니다.

예를 들어 KMeans의 0번 군집이 실제 레이블 0인 Business 카테고리를 의미하지는 않습니다.

따라서 각 군집에 포함된 기사 제목을 직접 확인하여 군집이 어떤 뉴스 주제를 나타내는지 분석했습니다.

```python
for cluster_id in range(6):
    print(f"Cluster {cluster_id}")

    samples = df[df["cluster"] == cluster_id]["title"].head(20)

    for title in samples:
        print("-", title)
```

특정 군집에서 다음과 같은 기사들이 반복적으로 나타난다면 해당 군집을 Sports 카테고리로 해석할 수 있습니다.

```text
- Team wins the championship final
- Player signs a new contract
- Coach announces the starting lineup
```

군집의 의미를 파악한 뒤 실제 제출 레이블과 연결했습니다.

```python
cluster_to_label = {
    0: 3,
    1: 4,
    2: 0,
    3: 5,
    4: 1,
    5: 2
}

submission["label"] = df["cluster"].map(cluster_to_label)
```

군집 번호와 레이블의 대응 관계는 실험 결과에 따라 달라질 수 있으므로, 매번 군집별 기사 내용을 확인한 뒤 매핑했습니다.

<br>

### 6.5 Scaling Experiments

임베딩 벡터에 스케일링을 적용했을 때 군집 성능이 달라지는지 비교했습니다.

| Experiment | Public Score |
|---|---:|
| Original MPNet Embedding | 0.782755 |
| StandardScaler | 0.786907 |
| MaxAbsScaler | 0.787730 |
| MinMaxScaler | 0.787992 |

스케일링을 통해 일부 성능 향상이 있었지만, 임베딩 모델을 변경했을 때만큼 큰 차이가 나타나지는 않았습니다.

문장 임베딩은 이미 의미적 거리 구조를 갖고 있기 때문에 스케일링 방식에 따라 기존의 벡터 공간이 달라질 수 있다는 점도 함께 고려했습니다.

<br>

### 6.6 Repeated Clustering Experiments

KMeans는 초기 중심점에 따라 서로 다른 군집 결과를 생성할 수 있습니다.

동일한 MPNet 임베딩을 기반으로 여러 군집 결과를 생성하고 다음 항목을 반복적으로 확인했습니다.

- 군집별 기사 개수
- 군집별 대표 기사
- 서로 혼합되는 뉴스 카테고리
- 군집과 실제 카테고리의 대응 관계
- Public Score 변화

단순히 Public Score가 가장 높은 결과만 선택하기보다, 군집별 기사 구성이 실제 뉴스 카테고리와 의미적으로 일관되게 대응하는지도 함께 검토했습니다.

<br>

## 7. Experiments

전체 제출 중 주요 성능 변화는 다음과 같습니다.

| Stage | Submission | Public Score |
|---|---|---:|
| Baseline | `baseline_submit.csv` | 0.467667 |
| Model comparison | `paraphrase-MiniLM-L12-v2.csv` | 0.616425 |
| Model comparison | `all-roberta-large-v1.csv` | 0.696113 |
| Model comparison | `paraphrase-MiniLM-L6-v2.csv` | 0.737111 |
| MPNet 적용 | `all-mpnet-base-v2.csv` | 0.782755 |
| StandardScaler | `all-mpnet-base-v2standard.csv` | 0.786907 |
| MaxAbsScaler | `all-mpnet-base-v2MaxAbs.csv` | 0.787730 |
| MinMaxScaler | `all-mpnet-base-v2MinMax.csv` | 0.787992 |
| 반복 실험 | `all-mpnet-base-v2_29.csv` | 0.797111 |
| 반복 실험 | `all-mpnet-base-v2_32.csv` | 0.797221 |
| Best Public | `all-mpnet-base-v2_36.csv` | **0.797515** |

### Score Progress

```text
Baseline
0.467667
    │
    ├─ Sentence embedding model comparison
    │       0.616425 → 0.696113 → 0.737111
    │
    ├─ all-mpnet-base-v2
    │       0.782755
    │
    ├─ Scaling and preprocessing experiments
    │       0.786907 → 0.787992
    │
    └─ Repeated clustering and cluster interpretation
            0.797515
```

Baseline 대비 최고 Public Score는 다음과 같이 향상되었습니다.

```text
0.467667 → 0.797515

Absolute Improvement: +0.329848
```

`all-mpnet-base-v2`를 처음 적용한 결과와 비교하면 후속 전처리 및 군집 실험을 통해 추가적인 성능 향상을 얻었습니다.

```text
0.782755 → 0.797515

Absolute Improvement: +0.014760
```

<br>

## 8. Results

| Metric | Result |
|---|---:|
| Best Public Score | **0.797515** |
| Private Score | **0.793410** |
| Final Rank | **8위** |
| Number of Submissions | **50회** |

Public과 Private 점수의 차이는 약 0.0041로, 최종 평가 데이터에서도 비교적 안정적인 성능을 유지했습니다.

```text
Public  : 0.797515
Private : 0.793410
Gap     : 0.004105
```

최종 모델은 별도의 정답 데이터나 복잡한 지도학습 모델 없이 다음 요소들을 활용했습니다.

- 데이터에 특화된 텍스트 전처리
- Sentence-BERT 기반 문장 임베딩
- KMeans 기반 비지도 군집화
- 군집별 대표 기사 분석
- 군집과 뉴스 카테고리 매핑
- 반복 실험을 통한 군집 결과 검증

<br>

## 9. My Contribution

본 프로젝트의 데이터 분석, 전처리, 모델링 및 결과 해석 과정을 개인으로 수행했습니다.

### Data Analysis

- 뉴스 기사 데이터 구조 분석
- 제목과 본문 결합 방식 설계
- 기사 데이터 내부의 HTML 및 반복 문자열 탐색
- 데이터 특성에 맞는 전처리 규칙 설계

### Modeling

- Sentence-Transformer 계열 임베딩 모델 비교
- `all-mpnet-base-v2` 기반 문장 임베딩 생성
- KMeans를 활용한 6개 뉴스 카테고리 군집화
- StandardScaler, MinMaxScaler, MaxAbsScaler 비교
- 여러 군집 결과에 대한 반복 실험

### Cluster Analysis

- 군집별 대표 기사 확인
- 군집의 의미적 특성 해석
- 군집과 실제 뉴스 카테고리 간 매핑
- 카테고리 혼합 사례 및 오분류 가능성 분석

### Experiment Management

- 총 50회의 제출 결과 비교
- 임베딩 모델, 전처리 및 군집 결과별 점수 기록
- Public Score와 군집 해석 결과를 함께 고려한 최종 결과 선정

<br>

## 10. Lessons Learned

### 1. 비지도 분류에서는 데이터 표현 방법이 중요하다

동일한 KMeans 알고리즘을 사용하더라도 문장 임베딩 모델에 따라 성능이 크게 달라졌습니다.

```text
Baseline                 : 0.467667
paraphrase-MiniLM-L6-v2 : 0.737111
all-mpnet-base-v2       : 0.782755
```

군집화 알고리즘을 복잡하게 변경하기 전에 데이터의 의미를 잘 보존하는 표현 방법을 선택하는 것이 중요하다는 점을 확인했습니다.

<br>

### 2. 전처리는 실제 데이터의 노이즈를 확인한 뒤 설계해야 한다

일반적인 특수문자 제거만으로는 충분하지 않았습니다.

뉴스 기사에 반복적으로 포함된 HTML 관련 문자열, 통신사 표기, URL 및 링크 표현 등을 직접 확인하고 제거하면서 데이터에 맞는 전처리 과정을 설계했습니다.

<br>

### 3. 군집 번호에는 의미가 없으며 해석 과정이 필요하다

KMeans의 군집 번호는 뉴스 카테고리 번호와 직접적으로 대응하지 않습니다.

군집별 기사들을 직접 확인하고 공통된 주제를 파악한 뒤 실제 뉴스 카테고리에 연결해야 했습니다.

이를 통해 비지도학습의 결과를 실제 문제의 레이블 체계로 변환하고 해석하는 과정을 경험했습니다.

<br>

### 4. KMeans 결과의 변동성을 관리해야 한다

초기 중심점에 따라 군집 경계와 성능이 달라질 수 있었습니다.

하나의 실행 결과만 사용하는 것이 아니라 여러 군집 결과를 비교하고, 점수뿐만 아니라 군집의 의미적 일관성까지 함께 검토했습니다.

<br>

### 5. 짧은 대회에서는 실험 우선순위를 정해야 한다

대회 기간이 짧았기 때문에 모든 방법을 시도할 수는 없었습니다.

성능에 미치는 영향이 클 것으로 예상되는 요소부터 순서대로 검증했습니다.

```text
임베딩 모델 선택
        ↓
입력 텍스트 및 전처리
        ↓
스케일링 방식 비교
        ↓
군집 결과 반복 검증
```

모델의 복잡도를 무작정 높이기보다, 각 실험에서 하나의 요소를 변경하고 결과를 비교하는 방식으로 접근했습니다.

<br>

## 11. Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | Sentence-Transformers |
| Embedding | all-mpnet-base-v2 |
| Machine Learning | Scikit-learn |
| Clustering | KMeans |
| Scaling | StandardScaler, MinMaxScaler, MaxAbsScaler |
| Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

<br>

## 12. Repository Structure

```text
뉴스 기사 레이블 복구 해커톤/
│
├── README.md
│
└── notebooks/
    ├── [Baseline]_SentenceBERT + KMeans.ipynb
    ├── all-mpnet-base-v2.ipynb
    ├── all-mpnet-base-v2standard.ipynb
    ├── all-mpnet-base-v2MinMax.ipynb
    ├── all-mpnet-base-v2MaxAbs.ipynb
    ├── all-mpnet-base-v2_11.ipynb
    ├── ...
    └── all-mpnet-base-v2_38.ipynb
```

<br>

## 13. Summary

본 프로젝트에서는 정답 레이블이 제공되지 않은 뉴스 기사 데이터를 대상으로 Sentence-BERT와 KMeans를 결합해 뉴스 카테고리 레이블을 복구했습니다.

여러 문장 임베딩 모델을 비교한 결과 `all-mpnet-base-v2`가 가장 높은 성능을 보였으며, 데이터에 특화된 전처리와 반복적인 군집 분석을 통해 Public Macro F1 Score를 0.467667에서 0.797515까지 개선했습니다.

최종적으로 Private Score 0.793410을 기록하며 대회를 8위로 마무리했습니다.

이 프로젝트를 통해 다음과 같은 역량을 발전시킬 수 있었습니다.

- 정답이 없는 데이터에서 문제를 해결하는 비지도학습 설계 역량
- 문장 임베딩을 활용한 텍스트 의미 표현 역량
- 군집 결과를 실제 레이블 체계로 해석하는 능력
- 데이터 특성에 맞는 전처리 규칙 설계 역량
- 제한된 기간 안에 실험 우선순위를 정하고 성능을 개선하는 역량

> 복잡한 모델을 적용하기 전에 데이터가 가진 의미를 적절하게 표현하고,  
> 모델이 생성한 결과를 문제의 맥락 안에서 해석하는 것이 중요하다는 점을 배웠습니다.

<br>

## Links

- [Competition Page](https://dacon.io/competitions/official/236159/overview/description)
- [Experiment Notebooks](./notebooks)
````
