# 온라인 채널 제품 판매량 예측

> **LG Aimers 3기 온라인 해커톤**  
> 제품 코드 중심의 판매 데이터를 실제 제품 의미와 소비 패턴 기반 Feature로 재구성하여 향후 판매량을 예측한 프로젝트입니다.

---

## Table of Contents

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
- [11. Repository Structure](#11-repository-structure)
- [12. Tech Stack](#12-tech-stack)

---

# 1. Project Overview

본 프로젝트는 **LG Aimers 3기 온라인 채널 제품 판매량 예측 AI 온라인 해커톤**을 대상으로 수행한 시계열 예측 프로젝트입니다.

대회의 목표는 온라인 쇼핑몰에서 판매되는 제품별 일별 판매 데이터를 기반으로, 향후 21일 동안의 제품별 판매량을 예측하는 것이었습니다.

기본 Baseline은 과거 판매량 데이터를 Sliding Window 형태로 구성한 뒤 LSTM 모델을 학습하는 방식이었습니다.  
그러나 제공 데이터는 제품명이 아닌 **제품 코드** 중심으로 구성되어 있었기 때문에, 모델이 실제 제품의 의미나 소비 패턴을 충분히 학습하기 어렵다는 한계가 있었습니다.

본 프로젝트에서는 모델 구조를 복잡하게 변경하기보다, **제품 코드에 담긴 의미를 해석하고 도메인 지식 기반 Feature를 추가하는 방향**으로 접근했습니다.

핵심 아이디어는 다음과 같습니다.

> 제품 ID를 단순한 코드로 두지 않고,  
> 실제 제품군, 브랜드, 가격, 소비 주기, 함께 사용되는 제품 관계, 요일 및 공휴일 정보로 확장한다.

이를 통해 Baseline 대비 약 **38.6% 성능 향상**을 달성하였고,  
최종적으로 **747팀 중 57위, 상위 약 7%**로 마무리했습니다.

---

# 2. Competition Overview

| 항목 | 내용 |
|---|---|
| Competition | 온라인 채널 제품 판매량 예측 AI 온라인 해커톤 |
| Program | LG Aimers 3기 |
| Host | LG AI Research / DACON |
| Period | 2023.08.01 ~ 2023.08.28 |
| Task | Time Series Forecasting |
| Objective | 제품별 향후 21일 판매량 예측 |
| Evaluation | Pseudo SFA / SFA |
| Team | 4인 팀 |
| Role | Team Leader |

<br/>

## 2.1 Task Description

온라인 판매 채널에서 수집된 제품별 일별 판매 데이터를 활용하여, 특정 제품이 미래 21일 동안 얼마나 판매될지를 예측하는 문제입니다.

즉, 각 제품 ID별로 다음과 같은 형태의 시계열 데이터가 주어집니다.

```text
Product ID
 ├── 2022-01-01 sales
 ├── 2022-01-02 sales
 ├── ...
 └── 2023-04-04 sales
```

모델은 이 과거 판매량 및 메타 정보를 활용하여 다음 기간의 판매량을 예측해야 합니다.

```text
2023-04-05 ~ 2023-04-25
```

<br/>

## 2.2 Evaluation Metric

본 대회는 **SFA(Sales Forecasting Accuracy)** 계열의 평가 지표를 사용했습니다.

대회 페이지에서는 평가 산식이 **Pseudo SFA(PSFA)**로 공개되었으며, Public Score는 테스트 데이터 일부, Private Score는 전체 테스트 데이터 기준으로 산출되었습니다.

본 프로젝트에서는 모델의 예측 판매량이 실제 판매량 패턴과 얼마나 유사한지를 높이는 것을 목표로 했습니다.

---

# 3. Dataset

본 대회에서 제공된 주요 데이터는 다음과 같습니다.

| File | Description |
|---|---|
| `train.csv` | 제품별 일별 판매량 |
| `sales.csv` | 제품별 일별 총 판매금액 |
| `brand_keyword_cnt.csv` | 브랜드별 연관 키워드 언급량 |
| `product_info.csv` | 일부 제품의 제품 특성 텍스트 |
| `sample_submission.csv` | 제출 양식 |

<br/>

## 3.1 train.csv

`train.csv`는 제품별 일별 판매량을 포함합니다.

기본 구조는 다음과 같습니다.

| Column | Description |
|---|---|
| `ID` | 판매되고 있는 고유 ID |
| `제품` | 제품 코드 |
| `대분류` | 제품 대분류 코드 |
| `중분류` | 제품 중분류 코드 |
| `소분류` | 제품 소분류 코드 |
| `브랜드` | 브랜드 코드 |
| `2022-01-01 ~ 2023-04-04` | 일별 판매량 |

특징적으로, 데이터는 일반적인 시계열 데이터처럼 `date` column이 따로 있는 long-format이 아니라, 날짜가 column으로 펼쳐진 wide-format 형태였습니다.

```text
ID | 제품 | 대분류 | 중분류 | 소분류 | 브랜드 | 2022-01-01 | 2022-01-02 | ...
```

따라서 Feature Engineering 과정에서는 여러 데이터셋을 날짜 기준으로 다루기 위해 wide-format과 long-format을 반복적으로 변환했습니다.

<br/>

## 3.2 sales.csv

`sales.csv`는 제품별 일별 총 판매금액 정보를 포함합니다.

판매량 데이터와 함께 활용하면 제품별 가격 흐름이나 가격 변화율을 구성할 수 있습니다.

본 프로젝트에서는 가격 관련 Feature를 만들기 위해 `sales.csv` 및 가격 관련 데이터를 활용했습니다.

<br/>

## 3.3 brand_keyword_cnt.csv

`brand_keyword_cnt.csv`는 브랜드별 연관 키워드 언급량을 날짜별로 제공한 데이터입니다.

이 데이터는 브랜드의 관심도 또는 검색량과 유사한 신호로 해석할 수 있습니다.

브랜드 언급량은 다음과 같은 흐름으로 가공했습니다.

```text
brand_keyword_cnt.csv
        │
        ▼
브랜드별 날짜 데이터 변환
        │
        ▼
제품 ID 기준으로 병합
        │
        ▼
브랜드 검색량 / 언급량 Feature 후보 생성
```

다만 최종 제출 모델에서는 모든 후보 Feature를 사용하지 않고, 성능과 안정성을 기준으로 일부 Feature를 제외했습니다.

<br/>

## 3.4 product_info.csv

`product_info.csv`는 제품 특성 텍스트를 포함하지만, 모든 제품 코드가 포함되어 있지는 않았습니다.

즉, `train.csv`에 존재하는 제품이 `product_info.csv`에 없거나, 반대로 `product_info.csv`에 존재하는 제품이 학습 데이터에 없는 경우가 있었습니다.

이로 인해 제품의 실제 의미를 파악하는 데 한계가 있었고, 본 프로젝트에서는 제품 코드를 직접 검토하여 실제 제품군과 소비 패턴을 해석하는 작업을 수행했습니다.

---

# 4. Motivation

## 4.1 Baseline의 한계

Baseline 모델은 기본적으로 과거 판매량을 Sliding Window 형태로 잘라 LSTM에 입력하는 방식이었습니다.

```text
Past Sales Sequence
        │
        ▼
LSTM
        │
        ▼
Future Sales Prediction
```

이 방식은 판매량 자체의 패턴은 학습할 수 있지만, 다음과 같은 정보는 충분히 반영하기 어렵습니다.

- 제품이 어떤 종류인지
- 어떤 브랜드인지
- 함께 구매될 가능성이 높은 제품군이 있는지
- 제품별 소비 속도가 다른지
- 특정 요일이나 공휴일에 판매량이 변하는지
- 가격 변화가 판매량에 영향을 주는지

즉, 모델은 `제품 코드`를 보지만, 해당 코드가 **치약인지, 샴푸인지, 면도용품인지** 알 수 없습니다.

<br/>

## 4.2 핵심 문제의식

본 프로젝트는 다음 질문에서 시작했습니다.

> 제품 코드를 단순한 식별자로만 두지 않고,  
> 실제 소비 맥락을 반영한 Feature로 바꿀 수 없을까?

이 질문을 바탕으로 모델 구조를 크게 변경하기보다, 데이터가 담고 있는 의미를 확장하는 방향으로 접근했습니다.

---

# 5. Project Pipeline

전체 프로젝트 흐름은 다음과 같습니다.

```text
Raw Data
  │
  ├── train.csv
  ├── sales.csv
  ├── brand_keyword_cnt.csv
  └── product_info.csv
  │
  ▼
Data Understanding
  │
  ├── 제품 코드 구조 확인
  ├── 대/중/소분류 코드 확인
  ├── 브랜드 코드 확인
  └── 제품 특성 텍스트 확인
  │
  ▼
Manual Product Interpretation
  │
  ├── 제품 코드별 실제 제품군 파악
  ├── 카테고리 재정리
  ├── 함께 사용하는 제품군 정의
  └── 소비주기 정의
  │
  ▼
Feature Engineering
  │
  ├── 판매량 변화율
  ├── 가격 변화율
  ├── 요일
  ├── 공휴일
  ├── 함께사용
  └── 소비주기
  │
  ▼
Sliding Window Dataset
  │
  ├── Input Window: 90 days
  └── Prediction Window: 21 days
  │
  ▼
LSTM Model
  │
  ▼
Submission
```

---

# 6. Feature Engineering

본 프로젝트의 핵심은 Feature Engineering입니다.

단순히 모델을 바꾸는 것이 아니라, 제품 코드 중심의 데이터를 실제 소비 맥락을 반영하는 형태로 재구성했습니다.

---

## 6.1 Product Category Mapping

### Problem

제공 데이터에는 제품명이 직접적으로 주어지지 않았고, 제품은 코드 형태로 표현되어 있었습니다.

```text
B002-XXXXX-XXXXX
```

이런 코드만으로는 모델이 제품의 실제 의미를 알 수 없습니다.

예를 들어, 모델 입장에서 다음 제품들이 어떤 차이를 갖는지 직접 알기 어렵습니다.

```text
제품 A = 치약
제품 B = 샴푸
제품 C = 면도기
제품 D = 세탁세제
```

### Approach

제품 코드와 제공된 제품 특성, 분류 코드를 바탕으로 각 제품의 실제 의미를 수작업으로 확인하고 정리했습니다.

이 과정에서 다음 정보를 중심으로 제품을 해석했습니다.

- 대분류
- 중분류
- 소분류
- 브랜드
- 제품 특성
- 실제 생활 속 사용 맥락

### Why It Matters

판매량 예측에서 제품의 의미는 매우 중요합니다.

예를 들어 치약과 칫솔은 서로 다른 제품이지만, 실제 소비 맥락에서는 함께 구매될 가능성이 있습니다.  
샴푸와 세제는 모두 반복 구매 상품이지만, 소비주기와 사용량은 다를 수 있습니다.

따라서 제품 코드를 단순 ID로 처리하는 대신, 제품의 실제 사용 맥락을 반영하는 Feature를 설계했습니다.

Implemented in:

```text
notebooks/product_category_mapping.ipynb
```

---

## 6.2 함께사용 Feature

### Idea

일부 제품은 단독으로 소비되기보다 다른 제품과 함께 사용되는 경향이 있습니다.

예를 들면 다음과 같습니다.

```text
치약     ↔ 칫솔
면도기   ↔ 쉐이빙폼
샴푸     ↔ 린스 / 트리트먼트
세탁세제 ↔ 섬유유연제
```

이러한 관계는 제품 판매량에 간접적으로 영향을 줄 수 있습니다.

### Design

제품의 소분류를 기준으로, 함께 사용될 가능성이 높은 제품군을 직접 정의했습니다.

예시는 다음과 같습니다.

| Group | Meaning |
|---|---|
| `teeth` | 구강 관리 관련 제품 |
| `shaving` | 면도 관련 제품 |
| `laundry` | 세탁 관련 제품 |
| `bath` | 목욕 / 바디케어 관련 제품 |
| `dish_wash` | 주방 세정 관련 제품 |
| `child_diaper` | 유아 위생 관련 제품 |
| `milk_powder` | 유아 식품 관련 제품 |

코드에서는 `함께사용`이라는 범주형 Feature를 생성한 뒤 Label Encoding을 적용하고, 날짜별 시계열 입력 형태로 확장했습니다.

```text
제품별 함께사용 그룹
        │
        ▼
Label Encoding
        │
        ▼
제품 ID × 날짜 형태로 확장
        │
        ▼
LSTM 입력 Feature
```

Implemented in:

```text
notebooks/product_category_mapping.ipynb
```

---

## 6.3 소비주기 Feature

### Idea

제품마다 소비되는 속도와 재구매 주기가 다릅니다.

예를 들어, 치약과 샴푸는 모두 반복 구매 상품이지만 사용 기간과 재구매 간격은 다르게 나타날 수 있습니다.  
또한 기저귀처럼 자주 소비되는 제품과, 면도기처럼 상대적으로 오래 사용하는 제품은 판매 패턴이 다릅니다.

### Design

본 프로젝트에서는 제품별 소비주기를 절대적인 날짜 단위로 추정하기보다, 제품군 간 상대적 소비 속도를 기준으로 나누었습니다.

```text
frequent
normal
not_frequent
```

즉, 다음과 같은 방식입니다.

- 자주 소비되는 제품군 → `frequent`
- 일반적인 소비 속도의 제품군 → `normal`
- 상대적으로 오래 사용하는 제품군 → `not_frequent`

이 기준은 실제 일상생활에서의 사용 경험과 제품 부피, 사용량, 재구매 가능성을 함께 고려하여 정의했습니다.

```text
제품 소분류
   │
   ▼
상대적 소비주기 정의
   │
   ▼
Label Encoding
   │
   ▼
제품 ID × 날짜 형태로 확장
   │
   ▼
LSTM 입력 Feature
```

Implemented in:

```text
notebooks/product_category_mapping.ipynb
```

---

## 6.4 가격 변화율 Feature

### Idea

가격은 판매량에 영향을 줄 수 있는 중요한 요인입니다.

특히 온라인 쇼핑몰에서는 할인, 프로모션, 가격 변동에 따라 판매량이 변할 수 있습니다.

### Processing

가격 관련 데이터는 다음 과정을 거쳐 사용했습니다.

```text
가격 / 판매금액 관련 데이터
        │
        ▼
0값 및 결측성 값 보정
        │
        ▼
일별 가격 흐름 구성
        │
        ▼
가격 변화율 계산
        │
        ▼
Scaling
        │
        ▼
LSTM 입력 Feature
```

코드에서는 가격 변화율을 다음과 같은 방식으로 계산했습니다.

```text
price_change_rate[t] =
    (price[t] - price[t-1]) / price[t-1]
```

분모가 0이 되는 경우와 `inf`, `NaN` 값은 0으로 처리하여 학습 안정성을 확보했습니다.

Implemented in:

```text
notebooks/feature_engineering.ipynb
notebooks/final_training_and_inference.ipynb
```

---

## 6.5 판매량 변화율 Feature

### Idea

판매량 자체뿐만 아니라 판매량이 증가하거나 감소하는 변화 방향도 중요한 정보입니다.

단순 판매량은 제품별 규모 차이가 크기 때문에, 변화율을 함께 입력하면 최근 추세를 더 잘 반영할 수 있습니다.

### Processing

판매량 변화율은 다음 방식으로 계산했습니다.

```text
sales_change_rate[t] =
    (sales[t] - sales[t-1]) / sales[t-1]
```

0으로 나누는 문제를 방지하기 위해 판매량이 0인 값에는 작은 값을 부여하고, 계산 이후 `inf`, `NaN` 값을 처리했습니다.

이 Feature는 최종 모델 입력에 포함되었습니다.

Implemented in:

```text
notebooks/feature_engineering.ipynb
notebooks/final_training_and_inference.ipynb
```

---

## 6.6 Calendar Feature

### Idea

온라인 판매량은 요일과 공휴일의 영향을 받을 수 있습니다.

예를 들어 주말, 명절, 대체공휴일, 특정 휴일에는 소비 행동이 달라질 수 있습니다.

### Features

다음 날짜 관련 Feature를 생성했습니다.

| Feature | Description |
|---|---|
| `weekday` | 요일 정보 |
| `holiday` | 주말 및 공휴일 여부 |

공휴일은 2022년과 2023년 주요 공휴일을 기준으로 직접 정의했습니다.

예시는 다음과 같습니다.

```text
2022-01-01  새해
2022-01-31 ~ 2022-02-02  설날
2022-09-09 ~ 2022-09-12  추석
2023-01-21 ~ 2023-01-24  설날 및 대체공휴일
```

최종 모델에서는 `weekday`, `holiday` Feature를 사용했습니다.

Implemented in:

```text
notebooks/feature_engineering.ipynb
notebooks/final_training_and_inference.ipynb
```

---

## 6.7 Brand and Price Candidate Features

브랜드 관련 데이터와 가격대 정보도 후보 Feature로 생성했습니다.

### Brand Keyword Count

`brand_keyword_cnt.csv`를 활용하여 브랜드별 연관 키워드 언급량을 날짜별로 정리했습니다.

```text
brand_keyword_cnt.csv
        │
        ▼
melt
        │
        ▼
브랜드 × 날짜 기준 집계
        │
        ▼
제품 ID 기준 병합
        │
        ▼
브랜드 언급량 Feature 후보 생성
```

또한 브랜드 언급량 변화율도 생성했습니다.

### Price Rank

가격대를 직접적인 가격 값으로 사용하는 대신, 가격을 구간화하고 순위화하는 방식도 실험했습니다.

```text
price
  │
  ▼
가격 단위 조정
  │
  ▼
price_rank 생성
  │
  ▼
제품별 가격대 Feature 후보
```

다만 최종 제출 모델에서는 모델 안정성과 성능을 기준으로 모든 후보 Feature를 사용하지는 않았습니다.

Implemented in:

```text
notebooks/brand_price_feature_engineering.ipynb
notebooks/feature_engineering.ipynb
```

---

## 6.8 Final Input Features

최종 LSTM 모델에는 총 7개의 시계열 Feature가 입력되었습니다.

| No. | Feature | Description |
|---:|---|---|
| 1 | `sales` | 일별 판매량 |
| 2 | `price_change_rate` | 가격 변화율 |
| 3 | `sales_change_rate` | 판매량 변화율 |
| 4 | `weekday` | 요일 |
| 5 | `holiday` | 주말 및 공휴일 여부 |
| 6 | `함께사용` | 함께 사용되는 제품군 |
| 7 | `소비주기` | 상대적 재구매 / 소비주기 |

입력 구조는 다음과 같습니다.

```text
For each Product ID:

Day t-89 ─ [sales, price_rate, sales_rate, weekday, holiday, co-use, cycle]
Day t-88 ─ [sales, price_rate, sales_rate, weekday, holiday, co-use, cycle]
...
Day t   ─ [sales, price_rate, sales_rate, weekday, holiday, co-use, cycle]

        │
        ▼

Predict next 21 days sales
```

---

# 7. Modeling

## 7.1 Model Choice

최종 모델은 Baseline 구조인 **LSTM**을 기반으로 했습니다.

복잡한 모델을 새로 설계하기보다, 동일한 모델 구조에서 입력 Feature를 확장하여 Feature Engineering의 효과를 확인하는 방향으로 접근했습니다.

<br/>

## 7.2 Sliding Window

학습 데이터는 다음과 같은 Sliding Window 방식으로 구성했습니다.

| Parameter | Value |
|---|---:|
| Train Window | 90 days |
| Predict Window | 21 days |
| Batch Size | 4096 |
| Epochs | 15 |
| Learning Rate | 1e-4 |
| Seed | 493 |

```text
Input:  Past 90 days
Output: Future 21 days
```

<br/>

## 7.3 LSTM Architecture

최종 모델 구조는 다음과 같습니다.

```text
Input Sequence
  shape = (batch, 90, 7)
        │
        ▼
LSTM
  input_size = 7
  hidden_size = 512
        │
        ▼
Last Hidden Output
        │
        ▼
Fully Connected Layer
        │
        ▼
ReLU
        │
        ▼
Dropout
        │
        ▼
Linear Output
        │
        ▼
21-day Forecast
```

모델의 출력은 제품별 향후 21일 판매량입니다.

Implemented in:

```text
notebooks/final_training_and_inference.ipynb
```

---

# 8. Results

| 항목 | 결과 |
|---|---:|
| Evaluation Metric | SFA / PSFA |
| Final Rank | 57 / 747 |
| Percentile | Top 7% |
| Improvement | 약 38.6% over Baseline |

본 프로젝트에서는 Baseline LSTM 구조를 유지하면서, 제품 코드 기반 데이터를 실제 소비 맥락을 반영하는 Feature로 확장하여 성능을 개선했습니다.

가장 중요한 성과는 단순히 모델을 바꾸는 것이 아니라, **모델이 학습할 수 있는 데이터의 의미를 확장했다는 점**입니다.

---

# 9. My Contribution

본 프로젝트는 4인 팀으로 진행되었으며, 저는 팀 리더로 참여했습니다.

주요 역할은 다음과 같습니다.

- 프로젝트 방향 설정
- 제품 코드 기반 데이터 구조 분석
- 제품 정보 수작업 검토 및 카테고리 정리
- 함께사용 Feature 설계
- 소비주기 Feature 설계
- 가격 변화율 및 판매량 변화율 Feature 생성
- 요일 및 공휴일 Feature 생성
- LSTM 입력 데이터 구성
- 최종 학습 및 제출 파일 생성
- 실험 결과 정리 및 프로젝트 문서화

특히 본 프로젝트의 핵심이었던 **도메인 지식 기반 Feature Engineering 과정 전반**을 주도했습니다.

---

# 10. Lessons Learned

## 10.1 모델보다 데이터의 의미가 중요할 수 있다

처음에는 모델 구조를 개선하는 것이 성능 향상의 핵심이라고 생각했습니다.

하지만 본 프로젝트를 진행하면서, 모델이 학습하는 데이터가 현실의 의미를 충분히 담고 있지 않다면 복잡한 모델을 사용하더라도 성능 개선에 한계가 있다는 점을 경험했습니다.

제품 코드를 단순한 식별자로 사용하는 대신, 실제 제품군, 소비주기, 함께 사용되는 관계, 가격 변화, 요일 및 공휴일 정보를 반영함으로써 모델이 더 많은 맥락을 학습할 수 있도록 만들었습니다.

<br/>

## 10.2 Feature Engineering은 문제 정의의 과정이다

Feature Engineering은 단순히 column을 추가하는 작업이 아니었습니다.

각 Feature는 다음 질문에 대한 답이었습니다.

```text
이 제품은 무엇인가?
이 제품은 어떤 제품과 함께 사용되는가?
이 제품은 얼마나 자주 재구매되는가?
가격 변화는 판매량에 영향을 줄 수 있는가?
특정 날짜의 소비 행동은 달라질 수 있는가?
```

이 질문에 답하는 과정에서, 데이터 분석은 단순한 모델 적용이 아니라 **문제를 다시 정의하고 데이터의 구조를 재설계하는 과정**이라는 점을 배웠습니다.

<br/>

## 10.3 현실의 소비 패턴을 모델 입력으로 바꾸는 경험

이 프로젝트에서 가장 의미 있었던 부분은 실제 생활 속 소비 패턴을 모델 입력 Feature로 변환했다는 점입니다.

예를 들어, 치약과 칫솔은 서로 다른 제품이지만 함께 사용되는 제품군으로 묶을 수 있습니다.  
샴푸와 면도기는 모두 반복 구매 상품이지만, 소비 주기는 다를 수 있습니다.  
공휴일이나 주말에는 온라인 구매 행동이 달라질 수 있습니다.

이러한 현실의 의미를 Feature로 구조화하는 과정이 최종 성능 향상으로 이어졌습니다.

---

## Notebook Description

| Notebook | Description |
|---|---|
| `product_category_mapping.ipynb` | 제품 코드 분석, 제품군 정리, 함께사용/소비주기 Feature 생성 |
| `brand_price_feature_engineering.ipynb` | 가격대, 가격 순위, 브랜드 검색량 관련 후보 Feature 생성 |
| `feature_engineering.ipynb` | 판매량 변화율, 가격 변화율, 날짜 Feature 등 모델 입력 Feature 생성 |
| `final_training_and_inference.ipynb` | 최종 LSTM 학습, 추론, 제출 파일 생성 |

---

# 11. Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Feature Engineering | Label Encoding, Sliding Window, Change Rate Features |
| Modeling | PyTorch, LSTM |
| Visualization | Matplotlib, Seaborn |
| Evaluation | SFA / PSFA |

---

# 12. Summary

본 프로젝트는 온라인 채널 판매량 예측 문제를 단순한 시계열 예측 문제가 아니라, **제품 코드에 숨겨진 실제 소비 맥락을 복원하는 문제**로 바라보았습니다.

제품 정보를 수작업으로 해석하고, 함께 사용되는 제품군과 소비주기, 가격 변화율, 판매량 변화율, 요일 및 공휴일 Feature를 설계함으로써 모델이 학습할 수 있는 정보를 확장했습니다.

결과적으로 Baseline 대비 약 **38.6% 성능 향상**을 달성했으며, 최종적으로 **747팀 중 57위, 상위 약 7%**를 기록했습니다.

이 프로젝트를 통해 데이터 분석에서 중요한 것은 단순히 모델을 적용하는 것이 아니라, **데이터가 현실의 의미를 담을 수 있도록 구조화하는 과정**임을 경험했습니다.
