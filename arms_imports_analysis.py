import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore')

df=pd.read_excel(r"C:\Users\srish\Downloads\dgft_arms.xlsx")
print(df)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = df.drop(['HS CODE', 'DESCRIPTION'], axis='columns')

column_sums = df.sum().reset_index()
column_sums.columns = ['Month', 'Total Value']
print(column_sums)

# ── Parse dates from column names like "APR-20 Value($)" ─────────────
column_sums['Date'] = pd.to_datetime(
    column_sums['Month'].str.replace(' Value($)', '', regex=False),
    format='%b-%y'
)
column_sums = column_sums.sort_values('Date').reset_index(drop=True)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(column_sums['Date'], column_sums['Total Value'], color='red', linewidth=1.2)

# X-axis: one tick per year only
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)

# Y-axis: show in millions
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))

# Gridlines + spines
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.grid(axis='x', which='minor', linestyle=':', alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)

# Optional: 12-month rolling average
# rolling = column_sums.set_index('Date')['Total Value'].rolling(12).mean()
# ax.plot(rolling, color='navy', linewidth=1.5, linestyle='--', label='12M MA')
# ax.legend()

ax.set_title('Arms Imports 2020–2026 (6-digit HS level)', fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('Import Value (USD)', fontsize=10)

plt.tight_layout()
plt.savefig('arms_imports_clean.png', dpi=150)
plt.show()

# df=df.drop(['HS CODE','DESCRIPTION'],axis='columns')
# print(df)


# column_sums = df.sum().reset_index()
# column_sums.columns = ['Month', 'Total Value']
# print(column_sums)


# plt.figure(figsize=(15,5))

# plt.plot(column_sums['Month'], column_sums['Total Value'], color='red')

# plt.xticks(rotation=45)
# plt.title("Arms imports over 2020-2026(6 digit level)")
# plt.xlabel("months")
# plt.ylabel("Values of imports")  
# plt.tight_layout()        

# plt.show()

print(column_sums)

from statsmodels.tsa.seasonal import seasonal_decompose



# Set date as index, ensure monthly frequency
from statsmodels.tsa.seasonal import seasonal_decompose

# ── Build time series ─────────────────────────────────────────────────
ts = column_sums.set_index('Date')['Total Value']
ts.index = pd.DatetimeIndex(ts.index, freq='MS')
print(ts)
# ── Decomposition ─────────────────────────────────────────────────────
result = seasonal_decompose(ts, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

axes[0].plot(result.observed, color='red', linewidth=1)
axes[0].set_ylabel('Observed')

axes[1].plot(result.trend, color='navy', linewidth=1.5)
axes[1].set_ylabel('Trend')

axes[2].plot(result.seasonal, color='green', linewidth=1)
axes[2].set_ylabel('Seasonal')

axes[3].plot(result.resid, color='gray', linewidth=1)
axes[3].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[3].set_ylabel('Residual')

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('Time Series Decomposition — Arms Imports 2020–2026', fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('decomposition.png', dpi=150)
plt.show()

# ── YoY Growth Rate ───────────────────────────────────────────────────
ts_df = ts.to_frame()
ts_df['YoY_%'] = ts.pct_change(12) * 100


print(ts_df.head(24))

# Print top 10 spike months precisely
top_spikes = column_sums.nlargest(10, 'Total Value')[['Date', 'Total Value']]
top_spikes['Total Value (Cr)'] = top_spikes['Total Value'] / 1e7
print(top_spikes.round(2))

from statsmodels.tsa.stattools import adfuller

result = adfuller(ts)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')


# If p-value > 0.05 → non-stationary → need differencing

from statsmodels.tsa.stattools import kpss

kpss_result = kpss(ts, regression='c', nlags='auto')

print("KPSS Test (H0: Series is Stationary)")
print(f"  KPSS Statistic : {kpss_result[0]:.4f}")
print(f"  p-value        : {kpss_result[1]:.4f}")
print(f"  Lags used      : {kpss_result[2]}")
print(f"  Critical Values:")
for key, val in kpss_result[3].items():
    print(f"    {key}: {val:.4f}")

if kpss_result[1] < 0.05:
    print("\n>> REJECT H0: Series is NON-STATIONARY")
else:
    print("\n>> FAIL TO REJECT H0: Series is STATIONARY")



# from statsmodels.tsa.statespace.sarimax import SARIMAX

# # SARIMA(p,d,q)(P,D,Q,s)
# # s=12 for monthly seasonal
# model = SARIMAX(ts,
#                 order=(1, 1, 1),           # non-seasonal: AR, diff, MA
#                 seasonal_order=(1, 1, 1, 12))  # seasonal: AR, diff, MA, period

# fit = model.fit(disp=False)
# print(fit.summary())

# # Forecast 12 months ahead
# forecast = fit.get_forecast(steps=12)
# forecast_mean = forecast.predicted_mean
# conf_int = forecast.conf_int()

# # Plot
# fig, ax = plt.subplots(figsize=(16, 5))
# ax.plot(ts, color='red', label='Observed')
# ax.plot(forecast_mean, color='navy', label='Forecast')
# ax.fill_between(conf_int.index,
#                 conf_int.iloc[:, 0],
#                 conf_int.iloc[:, 1],
#                 alpha=0.2, color='navy', label='95% CI')
# ax.legend()
# ax.set_title('SARIMA Forecast: Arms Imports 2026--27')
# plt.tight_layout()
# plt.show()

# pip install pmdarima
# import pmdarima as pm

# auto_model = pm.auto_arima(ts,
#                             seasonal=True,
#                             m=12,              # monthly seasonality
#                             stepwise=True,
#                             information_criterion='aic',
#                             trace=True)        # prints models tried

# print(auto_model.summary())
# forecast_auto = auto_model.predict(n_periods=12)
# print(auto_model.order)
# print(auto_model.seasonal_order)
# print(auto_model.summary())


# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
# fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# # ACF
# plot_acf(ts, lags=36, ax=axes[0])
# axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
# axes[0].grid(alpha=0.3)

# # PACF
# plot_pacf(ts, lags=36, ax=axes[1], method='ywm')
# axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
# axes[1].grid(alpha=0.3)

# plt.tight_layout()
# plt.show()




# Forecast next 12 months


# Create future dates
# future_dates = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1),
#                              periods=12,
#                              freq='MS')

# # Convert forecast to a Series
# forecast_series = pd.Series(forecast_auto, index=future_dates)

# # Plot
# plt.figure(figsize=(14,6))

# # Historical data
# plt.plot(ts.index, ts, label='Historical Data',
#          color='navy', linewidth=2)

# # Forecast
# plt.plot(forecast_series.index, forecast_series,
#          label='Auto ARIMA Forecast',
#          color='red', linewidth=2, linestyle='--')

# # Mark where forecasting starts
# plt.axvline(ts.index[-1], color='black',
#             linestyle=':', linewidth=1.5,
#             label='Forecast Start')

# plt.title('Auto ARIMA Forecast (Next 12 Months)', fontsize=16, fontweight='bold')
# plt.xlabel('Year')
# plt.ylabel('Total Value')
# plt.legend()
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()

# print(auto_model.order)          # check d
# print(auto_model.seasonal_order)

# print(ts)


# result = seasonal_decompose(ts, model='additive', period=12)
# result.seasonal.plot()
# plt.show()

# print(auto_model.order)
# print(auto_model.seasonal_order)
# print(auto_model.aic())


import numpy as np

# Step 1: Winsorize — cap values above 95th percentile
upper = ts.quantile(0.90)
ts_clean = ts.clip(upper=upper)

# Step 2: Re-run auto_arima on cleaned series
import pmdarima as pm

auto_model = pm.auto_arima(ts_clean,
                            seasonal=True,
                            m=12,
                            d=None,   # let it decide
                            D=1,      # force seasonal differencing
                            max_p=3, max_q=3,
                            max_P=2, max_Q=2,
                            trace=True,
                            stepwise=True,
                            information_criterion='aic')

print(auto_model.order)
print(auto_model.seasonal_order)

# Fit chosen order on ORIGINAL series
from statsmodels.tsa.statespace.sarimax import SARIMAX

p, d, q = auto_model.order
P, D, Q, s = auto_model.seasonal_order

model = SARIMAX(ts,
                order=(p, d, q),
                seasonal_order=(P, D, Q, 12))

fit = model.fit(disp=False)

# Forecast
forecast = fit.get_forecast(steps=12)
pred = forecast.predicted_mean
ci   = forecast.conf_int()

# Plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts, color='red', linewidth=1.2, label='Observed')
ax.plot(pred, color='navy', linewidth=1.5,
        linestyle='--', label='Forecast')
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1],
                alpha=0.2, color='navy', label='95% CI')
ax.axvline(ts.index[-1], color='gray', linestyle=':',
           linewidth=1, label='Forecast start')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend()
ax.set_title('SARIMA Forecast: Arms Imports 2026--27')
plt.tight_layout()
plt.show()



import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── Train-Test Split (holdout last 12 months) ─────────────────────────
train = ts[:-12]
test  = ts[-12:]

print(f"Training period: {train.index[0].strftime('%b %Y')} to "
      f"{train.index[-1].strftime('%b %Y')} ({len(train)} obs)")
print(f"Test period    : {test.index[0].strftime('%b %Y')} to "
      f"{test.index[-1].strftime('%b %Y')} ({len(test)} obs)")

# ── Refit Model on Train Only ─────────────────────────────────────────
# Use same order found after winsorisation
p, d, q = auto_model.order
P, D, Q, s = auto_model.seasonal_order

model_eval = SARIMAX(train,
                     order=(p, d, q),
                     seasonal_order=(P, D, Q, 12))
fit_eval = model_eval.fit(disp=False)

# ── Generate Forecasts ────────────────────────────────────────────────
forecast_eval = fit_eval.get_forecast(steps=12)
pred          = forecast_eval.predicted_mean
ci            = forecast_eval.conf_int()

# ── Accuracy Metrics ──────────────────────────────────────────────────
mae  = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = (np.abs((test - pred) / test) * 100).mean()
bias = (pred - test).mean()   # positive = overforecast

print(f"\n{'='*40}")
print(f"  Forecast Evaluation (12-month holdout)")
print(f"{'='*40}")
print(f"  MAE  : ${mae:>12,.0f}")
print(f"  RMSE : ${rmse:>12,.0f}")
print(f"  MAPE : {mape:>11.1f}%")
print(f"  Bias : ${bias:>12,.0f}  "
      f"({'overforecast' if bias > 0 else 'underforecast'})")
print(f"{'='*40}")

# ── Plot: Actual vs Predicted ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Panel 1: Full series + test window forecast
ax1 = axes[0]
ax1.plot(train, color='red', linewidth=1.2, label='Training data')
ax1.plot(test,  color='red', linewidth=1.2, linestyle='--',
         label='Actual (test)')
ax1.plot(pred,  color='navy', linewidth=1.5, linestyle='--',
         label='Forecast')
ax1.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                 alpha=0.2, color='navy', label='95% CI')
ax1.axvline(train.index[-1], color='gray', linestyle=':',
            linewidth=1, label='Forecast origin')
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.legend()
ax1.set_title('Forecast Evaluation: Actual vs Predicted (12-Month Holdout)',
              fontsize=12)

# Panel 2: Forecast errors
ax2 = axes[1]
errors = test - pred
ax2.bar(errors.index, errors, color=[
    'red' if e < 0 else 'navy' for e in errors],
    width=20, alpha=0.7)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhline(mae,  color='green', linestyle='--',
            linewidth=1, label=f'MAE = ${mae/1e6:.1f}M')
ax2.axhline(-mae, color='green', linestyle='--', linewidth=1)
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.legend()
ax2.set_title('Forecast Errors (Actual minus Predicted)', fontsize=12)
ax2.set_ylabel('Error (USD)')

plt.tight_layout()
plt.savefig('forecast_evaluation.png', dpi=150)
plt.show()

# ── Monthly Breakdown Table ───────────────────────────────────────────
eval_df = pd.DataFrame({
    'Actual ($)':    test.values,
    'Forecast ($)':  pred.values,
    'Error ($)':     (test - pred).values,
    'Error (%)':     ((test - pred) / test * 100).values
}, index=test.index.strftime('%b %Y'))

eval_df = eval_df.round(0)
print("\nMonthly Breakdown:")
print(eval_df.to_string())

