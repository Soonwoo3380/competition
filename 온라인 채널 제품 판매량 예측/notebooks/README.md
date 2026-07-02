# Online Channel Product Sales Forecasting

🌐 **Language** - 🇺🇸 **English (Current)** - 🇰🇷 [한국어](README_KR.md)

> **LG Aimers 3**
>
> Product Sales Forecasting using Domain-driven Feature Engineering

------------------------------------------------------------------------

## 📌 Overview

  Item          Description
  ------------- ----------------------------------
  Period        Aug. 2023
  Competition   LG Aimers 3
  Task          Online product sales forecasting
  Type          Time Series Forecasting
  Framework     PyTorch (LSTM)

------------------------------------------------------------------------

## 📖 Project Background

The provided dataset contained only **product codes**, making it
difficult for the model to learn meaningful product characteristics.

Instead of simply learning sales trends, this project focused on
enriching the data with **domain knowledge** through feature
engineering.

------------------------------------------------------------------------

## 🎯 Problem

The baseline model relied primarily on historical sales using a sliding
window.

However, actual sales are influenced by product category, brand,
repurchase cycle, co-purchased products, day of week, and holidays.

> **How can we provide the model with richer information beyond product
> IDs?**

------------------------------------------------------------------------

## 🚀 Approach

``` text
Product Code
      │
      ▼
Product Information Collection
      │
      ▼
Category Mapping
      │
      ▼
Domain Knowledge
      │
      ▼
Feature Engineering
      │
      ▼
LSTM Forecasting
```

### Notebooks

-   `01_product-category-mapping.ipynb`
-   `02_brand-price-feature-engineering.ipynb`
-   `03_feature-engineering.ipynb`
-   `04_final-training-and-inference.ipynb`

------------------------------------------------------------------------

## 🤖 Model

-   LSTM (PyTorch)
-   Sliding Window

------------------------------------------------------------------------

## 📈 Results

| Metric | Result |
|---------|-------:|
| Evaluation Metric | **SFA** |
| Public Score | **0.22639** |
| Private Score | **0.22946** |
| Rank | **31 / 740 (Top 4.2%)** |
| Improvement | **61.7% over Baseline** |

------------------------------------------------------------------------

## 💡 Key Takeaways

This project demonstrated that model performance depends not only on
algorithms but also on how well the data represents real-world meaning.
