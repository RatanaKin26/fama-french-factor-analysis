# Fama-French Factor Analysis

## Overview

This project replicates and extends the Fama-French asset pricing framework to evaluate how well different factor models explain portfolio returns.

The analysis compares four increasingly sophisticated models:

- CAPM
- Fama-French 3-Factor (FF3)
- Fama-French 5-Factor (FF5)
- Fama-French 5-Factor + Momentum (FF5+Mom)

The project demonstrates how additional risk factors improve explanatory power and provide a deeper understanding of portfolio performance.

---

## Business Question

Can portfolio returns be explained solely by market risk, or do additional factors such as size, value, profitability, investment, and momentum improve our understanding of asset returns?

This question is central to:

- Portfolio management
- Equity research
- Quantitative investing
- Risk management
- Performance attribution

---

## Methodology

### 1. Generate Factor Returns

Synthetic factor returns are generated using historical characteristics from the Ken French Data Library (1963–2023).

Factors include:

| Factor | Description |
|----------|-------------|
| Mkt-RF | Market Excess Return |
| SMB | Small Minus Big |
| HML | High Minus Low (Value) |
| RMW | Robust Minus Weak (Profitability) |
| CMA | Conservative Minus Aggressive (Investment) |
| UMD | Momentum |

---

### 2. Create Test Portfolios

Ten portfolios are generated with predefined exposures to the factors.

Examples include:

- Value
- Small Growth
- Momentum
- Quality
- Conservative
- Defensive Quality

Each portfolio contains unique factor sensitivities and idiosyncratic risk.

---

### 3. Exploratory Analysis

The project visualizes:

- Factor return distributions
- Cumulative factor performance
- Correlation structure between factors
- Rolling Sharpe ratios

---

### 4. Asset Pricing Regressions

Ordinary Least Squares (OLS) regressions are performed using:

#### CAPM

Ri - Rf = α + β(Market)

#### FF3

Ri - Rf = α + β(Market) + SMB + HML

#### FF5

Ri - Rf = α + β(Market) + SMB + HML + RMW + CMA

#### FF5 + Momentum

Ri - Rf = α + β(Market) + SMB + HML + RMW + CMA + UMD

For each portfolio the analysis estimates:

- Alpha
- Factor exposures (betas)
- t-statistics
- R²

---

### 5. Model Comparison

The project compares:

- Explanatory power (R²)
- Alpha reduction
- Improvement from adding factors

This demonstrates whether additional factors provide meaningful explanatory value.

---

### 6. Rolling Factor Exposures

Rolling 36-month regressions estimate time-varying factor exposures.

This helps illustrate:

- Factor stability
- Regime shifts
- Changes in portfolio characteristics over time

---

### 7. Risk-Return Analysis

Portfolio performance is evaluated using:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio

A risk-return scatter plot is created alongside the Capital Market Line.

---

## Key Findings

- CAPM explains a significant portion of portfolio returns but leaves substantial unexplained variation.
- Adding Size and Value factors improves model fit.
- FF5 further increases explanatory power by incorporating profitability and investment factors.
- Momentum provides additional explanatory value for momentum-oriented portfolios.
- Multi-factor models consistently achieve higher R² values than CAPM.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Statsmodels
- Matplotlib
- Seaborn

---

## Project Structure

```text
fama-french-factor-analysis/
│
├── run_analysis.py
├── README.md
├── requirements.txt
├── regression_results.csv
│
└── plots/
    ├── 01_cumulative_factor_returns.png
    ├── 02_factor_correlations.png
    ├── 03_rolling_sharpe_ratios.png
    ├── 04_r2_alpha_comparison.png
    ├── 05_avg_r2_by_model.png
    ├── 06_rolling_betas.png
    └── 07_risk_return_scatter.png
```

---

## How to Run

Clone the repository:

```bash
git clone <repository-url>
cd fama-french-factor-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python run_analysis.py
```

---

## Skills Demonstrated

- Financial Modeling
- Quantitative Finance
- Econometrics
- Time Series Analysis
- Factor Investing
- Regression Analysis
- Portfolio Analytics
- Data Visualization

---

## Author

Ratana Kin

Mathematical Economics | Data Science

Forecasting Analyst | Quantitative Analysis | Financial Analytics
