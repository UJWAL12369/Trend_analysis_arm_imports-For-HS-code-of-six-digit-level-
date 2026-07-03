# Trend Analysis and Forecasting of India's Arms Imports (HS Chapter 93)

## Overview

This project analyzes India's monthly imports of arms and ammunition under **ITC-HS Chapter 93** from **April 2020 to March 2026**. The analysis examines historical import trends, seasonal patterns, procurement spikes, and forecasts future imports using time series forecasting techniques.

The objective is to understand how geopolitical events, procurement cycles, and the Government of India's **Positive Indigenisation Lists (PILs)** have influenced arms imports and to develop a baseline forecasting model for future procurement.

---

## Dataset

- **Source:** Directorate General of Commercial Intelligence and Statistics (DGCIS)
- **Coverage:** April 2020 – March 2026
- **Frequency:** Monthly
- **Products:** 14 six-digit HS Codes under ITC-HS Chapter 93
- **Currency:** USD

The dataset contains monthly import values for:

- Revolvers and Pistols
- Sporting Firearms
- Shotguns
- Parts and Accessories
- Cartridges
- Bombs and Grenades
- Other Arms and Ammunition

---

## Project Objectives

- Analyze historical import trends.
- Study seasonal procurement behaviour.
- Identify abnormal procurement spikes.
- Examine the effect of Positive Indigenisation Lists (PIL).
- Forecast future monthly imports using SARIMA.
- Evaluate forecast accuracy using out-of-sample testing.

---

## Methodology

### Data Preprocessing

- Data cleaning
- Monthly aggregation
- Missing value handling
- Winsorization of extreme procurement spikes (90th percentile) for model selection

---

### Exploratory Data Analysis

- Time Series Visualization
- Trend Analysis
- Monthly Seasonality
- Additive Time Series Decomposition
- Year-on-Year Growth Analysis
- Procurement Spike Detection

---

### Stationarity Testing

Two statistical tests were performed:

- Augmented Dickey-Fuller (ADF)
- KPSS Test

Results confirmed that the series is stationary, eliminating the need for differencing before model estimation.

---

### Forecasting

Models considered:

- Auto ARIMA
- SARIMA

Initial Auto ARIMA selected a white-noise model because extreme procurement spikes obscured the autocorrelation structure.

To recover the underlying seasonal pattern:

- Extreme values were winsorized
- Grid Search was performed
- Final model selected using AIC

### Final Model

SARIMA

```
(2,0,1)(1,0,1,12)
```

---

## Forecast Evaluation

The final model was evaluated using a **12-month holdout test set**.

Performance metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Forecast Bias

The model successfully captured the underlying seasonal procurement pattern but, as expected, could not predict extraordinary procurement spikes caused by geopolitical events or emergency acquisitions.

---

## Visualizations

The project includes:

- Monthly Import Trend
- Time Series Decomposition
- Seasonal Component
- Residual Analysis
- Spike Analysis
- ACF & PACF Plots
- Forecast vs Actual Comparison
- Forecast Error Analysis
- 12-Month Future Forecast

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn
- SciPy

---

## Repository Structure

```
├── data/
│   └── Monthly_Imports.csv
│
├── notebooks/
│   └── Analysis.ipynb
│
├── figures/
│   ├── decomposition.png
│   ├── forecast.png
│   ├── acf_pacf.png
│   └── spikes.png
│
├── report/
│   └── Trend_Analysis_Arms_Imports.pdf
│
├── requirements.txt
│
└── README.md
```

---

## Key Findings

- Total imports exceeded **USD 671 million** during the study period.
- HS Code **930690 (Bombs, Grenades and Other Munitions)** accounted for over **64%** of total imports.
- Imports exhibited a strong annual seasonal procurement cycle.
- Major spikes were linked to:
  - Post-Galwan emergency procurement
  - Positive Indigenisation List (PIL) notifications
  - Russia–Ukraine conflict
  - Contract milestone payments
- SARIMA effectively modeled the baseline procurement pattern but not exogenous shocks.

---

## Limitations

- Procurement spikes are driven by external policy and geopolitical events that cannot be captured by a univariate time series model.
- Small sample size (72 monthly observations).
- Quantity information was unavailable; analysis is based solely on import value.

---

## Future Work

Potential extensions include:

- SARIMAX with exogenous policy variables
- Intervention Analysis
- Difference-in-Differences estimation for PIL impact
- Machine Learning forecasting models
- LSTM and Transformer-based forecasting
- Multivariate time series incorporating exchange rates and defence expenditure

---

## References

- Directorate General of Commercial Intelligence and Statistics (DGCIS)
- ITC-HS Chapter 93 Classification
- Statsmodels Documentation
- Box, Jenkins, Reinsel & Ljung (Time Series Analysis)

---

## Author

**Ujjwal Mittal**

M.A. Economics  
University of Delhi

---

## License

This project is intended for academic and research purposes.
