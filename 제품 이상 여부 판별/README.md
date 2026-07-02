# 제품 이상 여부 판별 프로젝트

> **LG Aimers 5기**  
> 디스플레이 제조 공정 데이터를 활용하여 제품의 이상 여부를 판별하는 이진 분류 프로젝트입니다.  
> 서로 다른 공정에서 생성된 센서 데이터를 공정 특성에 맞게 재구성하고, 데이터 불균형 문제를 고려한 다중 모델 Ensemble을 통해 AbNormal 제품 탐지 성능을 개선했습니다.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Competition Overview](#2-competition-overview)
- [3. Dataset](#3-dataset)
- [4. Motivation](#4-motivation)
- [5. Data Characteristics](#5-data-characteristics)
- [6. Project Pipeline](#6-project-pipeline)
- [7. Data Preprocessing](#7-data-preprocessing)
- [8. Process-level Data Reconstruction](#8-process-level-data-reconstruction)
- [9. Class Imbalance Handling](#9-class-imbalance-handling)
- [10. Modeling](#10-modeling)
- [11. Ensemble Strategy](#11-ensemble-strategy)
- [12. Results](#12-results)
- [13. My Contribution](#13-my-contribution)
- [14. Lessons Learned](#14-lessons-learned)
- [15. Tech Stack](#15-tech-stack)
- [16. Summary](#16-summary)
  
---

# 1. Project Overview

본 프로젝트는 **LG Aimers 5기 제품 이상 여부 판별 프로젝트**를 대상으로 수행한 제조 공정 데이터 기반 이진 분류 프로젝트입니다.

대회의 목표는 디스플레이 제조 공정에서 수집된 다양한 센서 및 공정 데이터를 활용하여, 최종 제품이 정상인지 이상 제품인지를 판별하는 모델을 개발하는 것이었습니다.

예측 대상은 다음 두 클래스입니다.

```text
Normal    : 정상 제품
AbNormal  : 이상 제품
```

본 프로젝트에서 가장 중요하게 본 문제는 다음과 같습니다.

> 제조 공정 데이터는 공정별 특성과 변수 구조가 다르며,  
> AbNormal 데이터가 매우 적은 불균형 데이터이기 때문에  
> 단순한 분류 모델만으로는 이상 제품을 안정적으로 탐지하기 어렵다.

따라서 본 프로젝트에서는 모델 자체를 단일하게 사용하는 것이 아니라,

- 공정별 데이터 구조 분석
- 결측 및 단일값 컬럼 제거
- 섞인 컬럼 복구
- 공정 간 일치성 Feature 생성
- Resin 계열 / Autoclave 계열 데이터 분리
- TomekLinks 기반 클래스 경계 샘플 정리
- Stratified K-Fold 기반 검증
- LightGBM, CatBoost, XGBoost Soft Voting Ensemble

을 통해 최종 예측 성능을 개선하고자 했습니다.

최종적으로 **Private Score 0.229457**, **740팀 중 31위(상위 4.2%)**를 기록했습니다.

---

# 2. Competition Overview

| 항목 | 내용 |
|---|---|
| Competition | 제품 이상 여부 판별 프로젝트 |
| Program | LG Aimers 5기 |
| Period | 2024.08.01 ~ 2024.08.27 |
| Task | Binary Classification |
| Objective | 제조 공정 데이터를 활용한 제품 이상 여부 판별 |
| Target | `Normal`, `AbNormal` |
| Evaluation Metric | F1 Score |
| Final Rank | 31 / 740 |
| Private Score | 0.229457 |
| Public Score | 0.226385 |
| Baseline Score | 0.14 |
| Improvement | 약 61.7% 향상 |
| Project Type | Individual Project |

<br/>

## 2.1 Task Description

디스플레이 제조 공정에서 생성된 데이터를 바탕으로 제품의 이상 여부를 예측하는 문제입니다.

데이터는 여러 공정 단계에서 수집된 센서값, 장비 정보, 작업 정보 등을 포함하고 있으며, 모델은 이 데이터를 입력받아 해당 제품이 `Normal`인지 `AbNormal`인지 예측해야 합니다.

```text
Manufacturing Process Data
          │
          ▼
Binary Classification Model
          │
          ▼
Normal / AbNormal
```

<br/>

## 2.2 Evaluation Metric

본 대회의 평가 지표는 **F1 Score**입니다.

제조 공정 이상 탐지 문제에서는 정상 제품보다 이상 제품의 비율이 훨씬 적기 때문에, 단순 정확도(Accuracy)는 적절한 평가 지표가 되기 어렵습니다.

예를 들어 전체 데이터의 대부분이 Normal일 경우, 모든 샘플을 Normal로 예측해도 Accuracy는 높게 나올 수 있습니다. 하지만 이 경우 실제로 중요한 AbNormal 제품을 탐지하지 못합니다.

따라서 본 프로젝트에서는 정밀도(Precision)와 재현율(Recall)을 함께 고려하는 F1 Score를 중심으로 성능을 개선하고자 했습니다.

```text
Precision = 예측한 AbNormal 중 실제 AbNormal 비율

Recall    = 실제 AbNormal 중 모델이 찾아낸 비율

F1 Score  = Precision과 Recall의 조화평균
```

---

# 3. Dataset

본 프로젝트에서는 대회에서 제공된 `train.csv`, `test.csv` 데이터를 사용했습니다.

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| `train.csv` | 40,506 | 464 | 학습용 제조 공정 데이터 및 target |
| `test.csv` | 17,361 | 465 | 제출용 제조 공정 데이터, `Set ID` 포함, target 미제공 |

<br/>

## 3.1 Train Data

학습 데이터는 총 **40,506개 샘플**, **464개 컬럼**으로 구성되어 있습니다.

Target 분포는 다음과 같습니다.

| Target | Count | Ratio |
|---|---:|---:|
| Normal | 38,156 | 94.20% |
| AbNormal | 2,350 | 5.80% |

즉, 학습 데이터는 정상 제품이 대부분을 차지하는 심한 클래스 불균형 구조를 가지고 있었습니다.

```text
Normal    ████████████████████████████████████████████████ 94.20%
AbNormal  ███ 5.80%
```

<br/>

## 3.2 Test Data

테스트 데이터는 총 **17,361개 샘플**, **465개 컬럼**으로 구성되어 있습니다.

Train 데이터와 비교했을 때 다음 차이가 있습니다.

- 제출용 식별자인 `Set ID` 컬럼 존재
- `target` 값은 비어 있음
- 모델 예측 결과를 `submission.csv`의 `target` 컬럼에 채워 제출

<br/>

## 3.3 Column Structure

컬럼명은 다음과 같이 공정 또는 원본 파일명을 suffix로 포함하는 형태였습니다.

```text
FeatureName_Dam
FeatureName_Fill1
FeatureName_Fill2
FeatureName_AutoClave
```

예시는 다음과 같습니다.

```text
Equipment_Dam
Workorder_Dam
Receip No Collect Result_Fill1
Production Qty Collect Result_Fill2
Chamber Temp. Judge Value_AutoClave
```

이러한 구조는 단일 테이블 안에 여러 공정의 데이터가 함께 포함되어 있음을 의미합니다.

---

# 4. Motivation

## 4.1 제조 공정 데이터의 어려움

제조 공정 데이터는 일반적인 정형 데이터와 달리 다음과 같은 특성을 가집니다.

1. 공정별 변수의 의미가 다르다.
2. 동일한 의미의 변수가 여러 공정에 반복적으로 등장한다.
3. 결측치가 많거나 모든 값이 동일한 컬럼이 존재한다.
4. 일부 컬럼에는 서로 다른 의미의 값이 섞여 있다.
5. 이상 제품 데이터가 매우 적다.
6. 정상 데이터 중심으로 학습하면 AbNormal 탐지 성능이 낮아질 수 있다.

따라서 단순히 모든 컬럼을 하나의 모델에 넣는 방식은 공정별 특성을 충분히 반영하지 못할 수 있다고 판단했습니다.

<br/>

## 4.2 핵심 문제의식

본 프로젝트는 다음 질문에서 출발했습니다.

> 서로 다른 제조 공정에서 생성된 변수를 하나의 테이블로만 볼 것이 아니라,  
> 공정 특성에 맞게 데이터를 재구성하면 모델이 더 잘 학습하지 않을까?

또한 클래스 불균형이 심한 상황에서는 모델이 Normal 중심으로 학습될 가능성이 높기 때문에, AbNormal을 더 안정적으로 탐지할 수 있도록 데이터와 모델링 과정을 설계하는 것이 중요했습니다.

---

# 5. Data Characteristics

## 5.1 Severe Class Imbalance

가장 큰 문제는 클래스 불균형이었습니다.

학습 데이터에서 AbNormal은 전체의 약 **5.8%**에 불과했습니다.

```text
Total Train Data: 40,506

Normal    : 38,156
AbNormal  :  2,350
```

이런 구조에서는 모델이 Normal을 많이 예측하는 방향으로 학습될 가능성이 높습니다.

따라서 본 프로젝트에서는 다음 방법을 사용했습니다.

- Stratified K-Fold
- TomekLinks
- Class Weight
- `scale_pos_weight`
- Soft Voting Ensemble

<br/>

## 5.2 High-dimensional Process Data

초기 데이터는 464개의 컬럼을 포함하고 있었습니다.

그러나 이 중 상당수는 다음과 같은 문제를 가지고 있었습니다.

- 모든 값이 결측치인 컬럼
- 결측치가 포함된 컬럼
- 모든 값이 동일한 컬럼
- Train에만 존재하거나 Test와 구조가 맞지 않는 컬럼
- 서로 다른 공정에 중복적으로 존재하는 컬럼

최종 노트북 기준으로 전처리 후 데이터 차원은 다음과 같이 정리되었습니다.

| Step | Train Shape | Test Shape |
|---|---:|---:|
| Raw Data | 40,506 × 464 | 17,361 × 465 |
| Missing Column Removal | -286 columns | -286 columns |
| Single-value Column Removal | -35 columns | -35 columns |
| Base Preprocessed Data | 40,506 × 146 | 17,361 × 146 |
| Resin Data | 40,506 × 130 | 17,361 × 130 |
| Autoclave Data | 40,506 × 14 | 17,361 × 14 |

<br/>

## 5.3 Process-level Structure

데이터는 크게 다음 공정 계열로 나누어 볼 수 있었습니다.

```text
Resin-related Process
 ├── Dam
 ├── Fill1
 └── Fill2

Autoclave-related Process
 └── AutoClave
```

최종 코드에서는 `autoclave` 포함 여부를 기준으로 데이터를 분리했습니다.

```text
All Features
  │
  ├── Features without "autoclave"  → Resin Data
  └── Features with "autoclave"     → Autoclave Data
```

---

# 6. Project Pipeline

전체 프로젝트 흐름은 다음과 같습니다.

```text
Raw Manufacturing Data
        │
        ├── train.csv
        └── test.csv
        │
        ▼
Column Name Normalization
        │
        ▼
Missing Column Removal
        │
        ▼
Single-value Column Removal
        │
        ▼
Mixed Column Correction
        │
        ▼
Process Consistency Feature Engineering
        │
        ├── receip_sameness
        ├── qty_sameness
        └── equipment_sameness
        │
        ▼
Process-level Data Split
        │
        ├── Resin Data
        └── Autoclave Data
        │
        ▼
Categorical Encoding
        │
        ▼
TomekLinks
        │
        ▼
Stratified K-Fold
        │
        ▼
Model Training
        │
        ├── LightGBM
        ├── CatBoost
        └── XGBoost
        │
        ▼
Soft Voting Ensemble
        │
        ▼
Submission
```

---

# 7. Data Preprocessing

## 7.1 Column Name Normalization

원본 컬럼명에는 공백, 괄호, 대소문자 등이 섞여 있었습니다.

예시는 다음과 같습니다.

```text
CURE END POSITION X Collect Result_Dam
CURE END POSITION X Unit Time_Dam
CURE END POSITION X Judge Value_Dam
```

전처리 과정에서는 컬럼명을 다음과 같이 통일했습니다.

```python
columns = columns.str.replace(' ', '_')
columns = columns.str.replace('(', '_')
columns = columns.str.replace(')', '_')
columns = columns.str.lower()
```

이를 통해 컬럼 접근과 Feature 처리 과정을 일관성 있게 만들었습니다.

---

## 7.2 Missing Column Removal

학습 데이터에서 결측치를 가진 컬럼을 확인한 뒤, 해당 컬럼을 Train/Test 양쪽에서 동일하게 제거했습니다.

```text
Columns with missing values in train data: 286
```

이 방식은 Train 기준으로 사용할 수 없는 컬럼을 제거하고, Test와 동일한 Feature 구조를 유지하기 위한 목적이었습니다.

```text
Train Columns
     │
     ▼
Find columns with missing values
     │
     ▼
Drop same columns from Train and Test
```

---

## 7.3 Single-value Column Removal

모든 값이 동일한 컬럼은 모델 학습에 유의미한 정보를 제공하지 않습니다.

따라서 Train 데이터 기준으로 unique 값이 1개뿐인 컬럼을 찾고, Train/Test 양쪽에서 제거했습니다.

```text
Single-value columns removed: 35
```

이 과정을 통해 불필요한 차원을 줄이고, 모델이 실제로 분류에 도움이 되는 Feature에 집중할 수 있도록 했습니다.

---

## 7.4 Mixed Column Correction

일부 컬럼에는 서로 다른 의미의 값이 섞여 있는 것으로 판단되는 패턴이 존재했습니다.

최종 코드에서는 다음 세 컬럼을 규칙 기반으로 보정했습니다.

```text
stage1_circle1_distance_speed_collect_result_dam
workmode_collect_result_dam
thickness_1_collect_result_dam
```

처리 아이디어는 다음과 같습니다.

```text
If value belongs to expected range:
    keep value
Else:
    treat as misplaced signal and move / correct
```

예를 들어 `stage1_circle1_distance_speed_collect_result_dam`에서 특정 기준보다 작은 값은 본래 의미와 다르게 섞인 값으로 보고 0으로 처리한 뒤, 관련 컬럼과 조합하여 값을 복구했습니다.

또한 `workmode_collect_result_dam`은 정상적으로 기대되는 값 범위를 기준으로 값의 위치를 보정했습니다.

이 과정은 단순 결측 처리보다 한 단계 더 나아가, 컬럼 내부 값의 의미를 해석하여 데이터 품질을 개선한 작업이었습니다.

---

# 8. Process-level Data Reconstruction

## 8.1 Resin Data

Resin 계열 데이터는 `autoclave`를 포함하지 않는 컬럼들로 구성했습니다.

```python
train_resin = train_data[
    [col for col in train_data.columns if 'autoclave' not in col]
]
```

이 데이터에는 주로 다음 공정이 포함됩니다.

```text
Dam
Fill1
Fill2
```

Resin 데이터에서는 동일한 의미를 가진 컬럼이 Dam, Fill1, Fill2에 반복적으로 존재했습니다.

예를 들어 다음과 같은 컬럼들이 있습니다.

```text
workorder_dam
workorder_fill1
workorder_fill2

model.suffix_dam
model.suffix_fill1
model.suffix_fill2
```

최종 코드에서는 중복된 의미의 컬럼을 정리했습니다.

```text
workorder_dam → workorder
model.suffix_dam → model.suffix

drop:
- workorder_fill1
- workorder_fill2
- model.suffix_fill1
- model.suffix_fill2
```

이는 여러 공정에 반복되는 동일 정보로 인해 모델이 불필요한 중복 패턴을 학습하지 않도록 하기 위한 처리입니다.

---

## 8.2 Autoclave Data

Autoclave 계열 데이터는 `autoclave`를 포함하는 컬럼만 별도로 추출했습니다.

```python
train_autoclave = train_data[
    [col for col in train_data.columns if 'autoclave' in col]
]
```

Autoclave 데이터에는 공정 자체의 변수뿐 아니라, Resin 계열 공정과의 연결성을 반영하기 위해 일부 일치성 Feature를 추가했습니다.

```text
Autoclave Features
    +
receip_sameness
qty_sameness
    +
target
```

Autoclave 데이터는 최종적으로 Resin 데이터와 별도 모델을 학습하는 데 사용되었습니다.

---

## 8.3 Process Consistency Features

공정 간 동일해야 할 값들이 실제로 일치하는지 확인하는 Feature를 생성했습니다.

생성한 Feature는 다음과 같습니다.

| Feature | Description |
|---|---|
| `receip_sameness` | Dam, Fill1, Fill2 공정의 Receip No가 모두 일치하는지 |
| `qty_sameness` | Dam, Fill1, Fill2 공정의 Production Qty가 모두 일치하는지 |
| `equipment_sameness` | Dam, Fill1, Fill2 공정의 장비 번호 suffix가 일치하는지 |

<br/>

### 8.3.1 receip_sameness

```text
receip_no_collect_result_dam
receip_no_collect_result_fill1
receip_no_collect_result_fill2
```

위 세 값이 모두 동일하면 `same`, 하나라도 다르면 `not_same`으로 정의했습니다.

Train 데이터 기준 분포는 다음과 같습니다.

| Value | Count |
|---|---:|
| same | 40,453 |
| not_same | 53 |

<br/>

### 8.3.2 qty_sameness

```text
production_qty_collect_result_dam
production_qty_collect_result_fill1
production_qty_collect_result_fill2
```

세 공정의 생산 수량 값이 모두 일치하는지 확인했습니다.

| Value | Count |
|---|---:|
| same | 40,417 |
| not_same | 89 |

<br/>

### 8.3.3 equipment_sameness

```text
equipment_dam
equipment_fill1
equipment_fill2
```

장비명 전체가 아니라, 장비 식별에 의미 있다고 판단한 뒤쪽 번호를 기준으로 일치 여부를 확인했습니다.

| Value | Count |
|---|---:|
| same | 40,472 |
| not_same | 34 |

<br/>

이 Feature들은 단순 센서값이 아니라 **공정 간 데이터 일관성**을 나타내는 Feature입니다.

제조 공정에서는 서로 다른 단계의 기록이 일치하지 않는 경우가 이상 징후와 관련될 수 있다고 판단했기 때문에, 이러한 일치성 정보를 모델에 반영했습니다.

---

# 9. Class Imbalance Handling

## 9.1 Target Imbalance

학습 데이터의 target 분포는 다음과 같습니다.

```text
Normal    : 38,156
AbNormal  :  2,350
```

AbNormal 비율은 약 5.8%로 매우 낮습니다.

이러한 상황에서 모델은 Normal을 예측하는 방향으로 쉽게 편향될 수 있습니다.

---

## 9.2 TomekLinks

본 프로젝트에서는 TomekLinks를 사용하여 클래스 경계에 위치한 샘플을 정리했습니다.

TomekLinks는 서로 다른 클래스에 속하면서 서로 가장 가까운 샘플 쌍을 찾고, 클래스 경계를 흐리게 만드는 샘플을 제거하는 방식입니다.

```text
Before TomekLinks
Normal / AbNormal boundary is noisy

After TomekLinks
Cleaner decision boundary
```

최종 코드에서는 Resin 데이터와 Autoclave 데이터 각각에 대해 TomekLinks를 적용했습니다.

```python
tomek = TomekLinks(sampling_strategy='auto')
X_tomek, y_tomek = tomek.fit_resample(X_t, y_t)
```

---

## 9.3 Class Weight

LightGBM과 CatBoost에서는 클래스 비율을 고려한 `class_weight`를 사용했습니다.

```python
class_weight = {
    0: count_0 / total,
    1: count_1 / total
}
```

XGBoost에서는 `scale_pos_weight`를 사용했습니다.

```python
scale_pos_weight =
    number_of_abnormal_samples / number_of_normal_samples
```

이러한 가중치 설정은 클래스 불균형 상황에서 모델이 특정 클래스에 과도하게 치우치지 않도록 하기 위한 목적이었습니다.

---

# 10. Modeling

본 프로젝트에서는 총 세 가지 Gradient Boosting 기반 모델을 사용했습니다.

```text
LightGBM
CatBoost
XGBoost
```

각 모델은 Resin 데이터와 Autoclave 데이터에 대해 별도로 학습되었습니다.

```text
Resin Data
  ├── LightGBM
  ├── CatBoost
  └── XGBoost

Autoclave Data
  ├── LightGBM
  ├── CatBoost
  └── XGBoost
```

---

## 10.1 Categorical Encoding

공정 데이터에는 장비명, 작업 지시 번호, 모델명 등 범주형 변수가 포함되어 있었습니다.

최종 코드에서는 object column에 대해 Label Encoding을 적용했습니다.

```text
Object Columns
      │
      ▼
LabelEncoder
      │
      ▼
Encoded Categorical Features
```

Test 데이터에 Train에서 보지 못한 새로운 category가 존재할 수 있기 때문에, Test의 unique label이 encoder class에 없을 경우 encoder class에 추가하는 방식으로 처리했습니다.

---

## 10.2 Stratified K-Fold

클래스 불균형이 심한 데이터이기 때문에, 단순 K-Fold를 사용하면 fold별 target 비율이 달라질 수 있습니다.

따라서 Stratified K-Fold를 사용하여 각 fold에서 Normal과 AbNormal 비율이 최대한 유지되도록 했습니다.

| Parameter | Value |
|---|---:|
| n_splits | 6 |
| shuffle | True |
| random_state | 18 |

```python
skf = StratifiedKFold(
    n_splits=6,
    shuffle=True,
    random_state=18
)
```

---

## 10.3 LightGBM

LightGBM은 대규모 정형 데이터에서 빠르고 효율적인 Gradient Boosting 모델입니다.

최종 코드에서는 다음 설정을 사용했습니다.

| Parameter | Value |
|---|---:|
| objective | binary |
| n_estimators | 5000 |
| random_state | 18 |
| early_stopping_rounds | 5 |

```python
LGBMClassifier(
    objective='binary',
    n_estimators=5000,
    class_weight=class_weight,
    random_state=18
)
```

---

## 10.4 CatBoost

CatBoost는 범주형 변수를 다루는 데 강점이 있는 Gradient Boosting 모델입니다.

본 프로젝트에서는 Label Encoding 후 object column을 category로 변환하고, `cat_features`로 전달했습니다.

| Parameter | Value |
|---|---:|
| class_weights | class_weight |
| random_state | 18 |
| early_stopping_rounds | 5 |

```python
CatBoostClassifier(
    class_weights=class_weight,
    cat_features=obj_cols,
    random_state=18
)
```

---

## 10.5 XGBoost

XGBoost는 정형 데이터 분류 문제에서 강력한 성능을 보이는 Gradient Boosting 모델입니다.

최종 코드에서는 categorical feature 사용과 클래스 불균형 보정을 함께 적용했습니다.

| Parameter | Value |
|---|---:|
| enable_categorical | True |
| n_estimators | 5000 |
| scale_pos_weight | class imbalance ratio |
| early_stopping_rounds | 5 |
| random_state | 18 |

```python
XGBClassifier(
    enable_categorical=True,
    n_estimators=5000,
    scale_pos_weight=...,
    early_stopping_rounds=5,
    random_state=18
)
```

---

# 11. Ensemble Strategy

## 11.1 Why Ensemble?

제조 공정 데이터는 변수 수가 많고, 각 공정별 데이터의 특성이 다르기 때문에 단일 모델의 예측에 의존할 경우 특정 패턴에 과적합될 가능성이 있습니다.

따라서 서로 다른 모델의 예측 확률을 평균내는 Soft Voting Ensemble을 적용했습니다.

```text
LightGBM Probability
        │
CatBoost Probability
        │
XGBoost Probability
        │
        ▼
Average Probability
        │
        ▼
Final Prediction
```

---

## 11.2 Probability Averaging

각 모델은 class별 확률값을 출력합니다.

```text
P(AbNormal), P(Normal)
```

최종 코드에서는 다음 예측 결과를 모두 누적했습니다.

```text
3 models × 2 process datasets × 6 folds
```

즉, 총 36개의 확률 예측을 평균낸 뒤 최종 class를 결정했습니다.

```python
pred_proba = zeros((17361, 2))

for elem in cat_pred_list + lgb_pred_list + xgb_pred_list:
    pred_proba += elem

final_proba = pred_proba / ((kfold_num * 2) * 3)
```

최종 class는 더 높은 확률을 가진 class로 결정했습니다.

```python
if P(AbNormal) > P(Normal):
    target = 'AbNormal'
else:
    target = 'Normal'
```

---

# 12. Results

| 항목 | 결과 |
|---|---:|
| Evaluation Metric | F1 Score |
| Public Score | 0.226385 |
| Private Score | 0.229457 |
| Final Rank | 31 / 740 |
| Percentile | Top 4.2% |
| Baseline Score | 0.14 |
| Improvement | 약 61.7% |

<br/>

## 12.1 Result Interpretation

본 프로젝트는 단순히 여러 모델을 적용한 것이 아니라, 제조 공정 데이터의 구조적 특성을 반영하여 데이터를 재구성한 뒤 Ensemble을 적용했습니다.

성능 개선에 기여한 핵심 요소는 다음과 같습니다.

- 결측 및 단일값 컬럼 제거를 통한 노이즈 감소
- 섞인 컬럼 복구를 통한 데이터 품질 개선
- 공정 간 일치성 Feature 생성
- Resin / Autoclave 데이터 분리
- TomekLinks 기반 클래스 경계 정리
- Stratified K-Fold 기반 안정적인 검증
- LightGBM, CatBoost, XGBoost Soft Voting Ensemble

---

# 13. My Contribution

본 프로젝트는 개인 프로젝트로 수행했습니다.

주요 수행 내용은 다음과 같습니다.

- 대회 문제 정의 및 데이터 구조 분석
- 제조 공정 컬럼 구조 파악
- 컬럼명 정규화
- 결측 컬럼 및 단일값 컬럼 제거
- 섞인 컬럼의 규칙 기반 복구
- Resin / Autoclave 공정 데이터 분리
- 공정 간 일치성 Feature 설계
  - `receip_sameness`
  - `qty_sameness`
  - `equipment_sameness`
- 범주형 변수 Label Encoding
- TomekLinks 기반 불균형 데이터 처리
- Stratified K-Fold 검증 구조 설계
- LightGBM 모델 학습
- CatBoost 모델 학습
- XGBoost 모델 학습
- Soft Voting Ensemble 구현
- 최종 제출 파일 생성
- 실험 결과 정리 및 GitHub 포트폴리오 문서화

---

# 14. Lessons Learned

## 14.1 제조 공정 데이터는 하나의 테이블로만 보면 안 된다

이 프로젝트를 통해 제조 공정 데이터는 단순한 정형 데이터가 아니라, 여러 공정에서 발생한 정보가 하나의 테이블에 결합된 구조라는 점을 경험했습니다.

처음에는 모든 컬럼을 하나의 입력 데이터로 처리할 수 있다고 생각했지만, 실제로는 Dam, Fill1, Fill2, Autoclave 등 공정마다 변수의 의미와 구조가 달랐습니다.

따라서 모델 성능을 높이기 위해서는 단순히 알고리즘을 변경하는 것보다, 공정의 구조를 이해하고 데이터를 학습 가능한 형태로 재구성하는 과정이 중요했습니다.

<br/>

## 14.2 이상 탐지 문제에서는 불균형 처리가 핵심이다

학습 데이터에서 AbNormal은 약 5.8%에 불과했습니다.

이런 상황에서는 모델이 Normal 중심으로 학습되기 쉽고, Accuracy는 높아도 실제로 중요한 AbNormal을 탐지하지 못할 수 있습니다.

따라서 F1 Score를 기준으로 모델을 평가하고, TomekLinks, class weight, Stratified K-Fold, Soft Voting Ensemble을 함께 적용했습니다.

이를 통해 이상 제품 탐지 문제에서는 **정확도보다 minority class를 얼마나 안정적으로 탐지하는가**가 중요하다는 점을 경험했습니다.

<br/>

## 14.3 Feature Engineering은 데이터 품질을 해석하는 과정이다

본 프로젝트에서 Feature Engineering은 새로운 변수를 많이 만드는 작업이라기보다, 제조 공정 데이터의 품질과 구조를 해석하는 과정에 가까웠습니다.

특히 다음 작업은 단순한 기술적 전처리가 아니라 문제를 이해하는 과정이었습니다.

```text
결측 컬럼 제거
단일값 컬럼 제거
섞인 컬럼 복구
공정 간 일치성 Feature 생성
공정별 데이터 분리
```

즉, 모델 성능은 알고리즘 자체뿐 아니라 데이터가 공정의 의미를 얼마나 잘 반영하도록 정리되었는지에 크게 영향을 받는다는 점을 배웠습니다.

<br/>

## 14.4 Ensemble은 다양한 관점을 결합하는 방법이다

LightGBM, CatBoost, XGBoost는 모두 Gradient Boosting 계열 모델이지만, 학습 방식과 범주형 변수 처리 방식, split 전략 등이 다릅니다.

본 프로젝트에서는 세 모델의 확률값을 평균내는 Soft Voting Ensemble을 통해 특정 모델의 편향을 줄이고, 보다 안정적인 예측을 만들고자 했습니다.

특히 Resin 데이터와 Autoclave 데이터를 따로 학습한 뒤 다시 결합하는 방식은, 서로 다른 공정 정보를 모델링 관점에서 분리하고 최종 판단 단계에서 통합하는 구조라고 볼 수 있습니다.

---

## Notebook Description

| Notebook | Description |
|---|---|
| `final_ensemble_training_and_inference.ipynb` | 데이터 전처리, 공정별 데이터 분리, TomekLinks, LightGBM/CatBoost/XGBoost 학습, Soft Voting Ensemble, 최종 제출 파일 생성 |

---

# 15. Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Preprocessing | LabelEncoder, Column Filtering, Feature Reconstruction |
| Imbalance Handling | TomekLinks, Class Weight |
| Validation | Stratified K-Fold |
| Modeling | LightGBM, CatBoost, XGBoost |
| Ensemble | Soft Voting |
| Evaluation | F1 Score |
| Visualization | Matplotlib, Seaborn |

---

# 16. Summary

본 프로젝트는 제조 공정 데이터를 활용한 제품 이상 여부 판별 문제를 단순한 이진 분류 문제가 아니라, **공정별 데이터 구조를 이해하고 재구성하는 문제**로 바라보았습니다.

원본 데이터는 여러 공정의 센서값과 작업 정보가 하나의 테이블에 결합되어 있었고, AbNormal 데이터가 매우 적은 불균형 구조를 가지고 있었습니다.

이를 해결하기 위해 결측 및 단일값 컬럼을 제거하고, 섞인 컬럼을 복구하며, 공정 간 일치성 Feature를 생성했습니다. 이후 Resin 계열 데이터와 Autoclave 계열 데이터를 분리하여 각각 LightGBM, CatBoost, XGBoost 모델을 학습하고, 최종적으로 Soft Voting Ensemble을 적용했습니다.

그 결과 **Private Score 0.229457**, **740팀 중 31위(상위 4.2%)**를 기록했으며, Baseline 대비 약 **61.7% 성능 향상**을 달성했습니다.

이 프로젝트를 통해 제조 데이터 분석에서 중요한 것은 단순히 모델을 적용하는 것이 아니라, **공정의 의미를 이해하고 데이터 구조를 학습 가능한 형태로 재정의하는 과정**임을 경험했습니다.
