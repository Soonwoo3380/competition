# 온라인 채널 제품 판매량 예측

> **LG Aimers 3기**
>
> 도메인 지식을 활용한 Feature Engineering 기반 온라인 채널 제품 판매량 예측

---

# 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 기간 | 2023.08 |
| 주최 | LG AI Research |
| 과제 | 온라인 채널 제품 판매량 예측 |
| 문제 유형 | Time Series Forecasting |
| 모델 | LSTM (PyTorch) |
| 평가 지표 | SFA (Sales Forecasting Accuracy) |

---

# 📖 프로젝트 배경

기본 데이터는 제품명이 아닌 **제품 코드(Product ID)** 만 제공되었습니다.

이러한 데이터만으로는 모델이 제품의 특성을 이해하기 어려우며,

- 어떤 제품인지
- 어떤 브랜드인지
- 어떤 제품과 함께 구매되는지
- 얼마나 자주 재구매되는지

와 같은 실제 소비 패턴을 반영하기 어렵다는 문제가 있었습니다.

본 프로젝트에서는 **모델을 변경하기보다 데이터의 의미를 확장하는 Feature Engineering**에 집중하여 성능을 개선하고자 하였습니다.

---

# 🎯 문제 정의

기존 Baseline은 과거 판매량을 기반으로 판매량을 예측했습니다.

하지만 실제 판매량은 단순한 시계열 정보뿐 아니라

- 제품 종류
- 브랜드
- 소비 주기
- 함께 구매되는 제품
- 요일 및 공휴일

등 다양한 요인의 영향을 받습니다.

따라서 제품 코드에 실제 도메인 정보를 부여하여 모델이 학습할 수 있는 정보를 확장하는 것을 목표로 하였습니다.

---

# 🚀 접근 방법

```text
제품 코드
      │
      ▼
제품 정보 조사
      │
      ▼
카테고리 분류
      │
      ▼
도메인 지식 반영
      │
      ▼
Feature Engineering
      │
      ▼
LSTM 기반 판매량 예측
```

---

# 🛠 Feature Engineering

### 1. 제품 카테고리 구축

제품 코드를 직접 조사하여

- 대분류
- 중분류
- 소분류
- 브랜드

정보를 수작업으로 구축하였습니다.

Notebook

`01_product-category-mapping.ipynb`

---

### 2. 브랜드 정보 활용

브랜드별

- 가격대
- 검색량

등의 정보를 활용하여 새로운 Feature를 생성하였습니다.

Notebook

`02_brand-price-feature-engineering.ipynb`

---

### 3. 소비 패턴 반영

실제 구매 패턴을 반영하기 위해

- 함께 사용하는 제품
- 재구매 주기
- 요일
- 공휴일

등의 Feature를 추가하였습니다.

Notebook

`03_feature-engineering.ipynb`

---

### 4. 최종 모델 학습

생성한 Feature를 활용하여

LSTM 기반 판매량 예측 모델을 학습하고 최종 제출 파일을 생성하였습니다.

Notebook

`04_final-training-and-inference.ipynb`

---

# 📈 결과

| 항목 | 결과 |
|------|------:|
| 평가 지표 | **SFA** |
| Public Score | **0.22639** |
| Private Score | **0.22946** |
| 최종 순위 | **31 / 740 (상위 4.2%)** |
| Baseline 대비 | **약 61.7% 성능 향상** |

---

# 💡 프로젝트를 통해 얻은 점

이번 프로젝트를 통해 **모델의 성능은 알고리즘 자체보다 데이터가 현실의 의미를 얼마나 잘 반영하는지에 크게 좌우된다**는 점을 경험했습니다.

제품 코드를 실제 제품 정보로 재구성하고, 도메인 지식을 Feature로 설계함으로써 모델이 학습할 수 있는 정보를 확장할 수 있었으며, 이를 통해 Baseline 대비 의미 있는 성능 향상을 달성했습니다.

---

# 📂 Repository Structure

```text
.
├── notebooks
│   ├── 01_product-category-mapping.ipynb
│   ├── 02_brand-price-feature-engineering.ipynb
│   ├── 03_feature-engineering.ipynb
│   └── 04_final-training-and-inference.ipynb
├── data
├── README.md
└── README_KR.md
```

---

# 🛠 Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- LSTM
- Feature Engineering
