# 제품 이상 여부 판별

> **LG Aimers 5기**
>
> 제조 공정 데이터를 활용한 제품 이상 여부 판별

---

# 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 기간 | 2024.08 |
| 주최 | LG AI Research |
| 과제 | 제품 이상 여부 판별 |
| 문제 유형 | Binary Classification |
| 모델 | LightGBM, CatBoost, XGBoost Ensemble |
| 평가 지표 | F1 Score |

---

# 📖 프로젝트 배경

제조 공정 데이터는 서로 다른 공정에서 생성되며,

- 동일한 의미의 변수 중복
- 결측치
- 이상치
- 심한 클래스 불균형

등의 문제를 포함하고 있었습니다.

또한 정상 제품의 비율이 매우 높아
불량 제품을 안정적으로 예측하기 어려웠습니다.

---

# 🎯 문제 정의

Baseline 모델은 공정 특성을 충분히 반영하지 못해
성능 향상에 한계가 있었습니다.

본 프로젝트에서는

- 공정별 데이터 구조화
- 데이터 품질 개선
- 데이터 불균형 완화
- Ensemble

을 통해 일반화 성능 향상을 목표로 하였습니다.

---

# 🚀 접근 방법

```text
Manufacturing Process Data
          │
          ▼
결측치 · 이상치 처리
          │
          ▼
공정별 데이터 분리
          │
          ▼
Feature Engineering
          │
          ▼
LightGBM
CatBoost
XGBoost
          │
          ▼
Soft Voting Ensemble
          │
          ▼
AbNormal / Normal 예측
```

---

# 🛠 데이터 전처리

### 1. 데이터 품질 개선

- 결측치 처리
- 이상치 처리
- 중복 변수 정리

---

### 2. 공정별 데이터 구조화

서로 다른 제조 공정 데이터를 분리하여

각 공정의 특성에 맞는 모델을 학습하도록 구성하였습니다.

---

### 3. 데이터 불균형 완화

정상 데이터가 대부분을 차지하는 문제를 해결하기 위해

- Tomek Links

를 적용하여 데이터 품질을 개선하였습니다.

---

# 🤖 모델

각 공정에 대해

- LightGBM
- CatBoost
- XGBoost

모델을 학습한 뒤

Soft Voting Ensemble을 통해 최종 예측을 수행하였습니다.

---

# 📈 결과

| 항목 | 결과 |
|------|------:|
| 평가 지표 | **F1 Score** |
| Public Score | **0.22639** |
| Private Score | **0.22946** |
| 최종 순위 | **31 / 740 (상위 4.2%)** |
| Baseline 대비 | **약 61.7% 성능 향상** |

---

# 💡 프로젝트를 통해 얻은 점

이번 프로젝트를 통해

**모델의 성능은 단순히 알고리즘을 변경하는 것이 아니라 공정 특성에 맞게 데이터를 재구성하는 과정에서도 크게 향상될 수 있다는 점**을 경험했습니다.

또한 제조 공정 데이터를 구조화하고 다양한 모델을 결합하는 Ensemble 전략을 통해 보다 안정적인 일반화 성능을 확보할 수 있었습니다.

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- CatBoost
- XGBoost
- Imbalanced-learn
- Ensemble Learning
