#!/usr/bin/env python3
"""
Fama-French Factor Model: CAPM → FF3 → FF5 → FF5+Momentum
===========================================================
Replicates and extends the Fama-French asset pricing framework.
Factor returns are synthetically generated, calibrated to historical
statistics from the Ken French Data Library (1963–2023).

Models: CAPM | FF3 | FF5 | FF5+Momentum
Tests:  10 portfolios with known factor exposures
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)
np.random.seed(42)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
colors = sns.color_palette('tab10')

# ============================================================
# 1. GENERATE FACTOR RETURNS
#    Calibrated to Ken French Data Library (1963–2023)
#    Monthly means and volatilities
# ============================================================
print("STEP 1: Generating Factor Returns (calibrated to 1963–2023)...")

dates = pd.date_range('1963-01', periods=732, freq='ME')  # 61 years monthly
T = len(dates)

# Monthly means and stds (calibrated to historical data)
#         Mkt-RF  SMB    HML    RMW    CMA
means = np.array([0.52,  0.18,  0.27,  0.25,  0.27]) / 100
stds  = np.array([4.50,  3.10,  3.10,  2.30,  2.00]) / 100

# Correlation matrix (from historical data)
corr = np.array([
    [ 1.00,  0.27, -0.34, -0.23, -0.38],
    [ 0.27,  1.00, -0.06, -0.38, -0.12],
    [-0.34, -0.06,  1.00,  0.13,  0.70],
    [-0.23, -0.38,  0.13,  1.00, -0.09],
    [-0.38, -0.12,  0.70, -0.09,  1.00],
])
cov = np.diag(stds) @ corr @ np.diag(stds)

# Draw correlated factor returns
raw = np.random.multivariate_normal(means, cov, T)

# Risk-free rate: realistic time-varying path
rf = np.full(T, 0.40 / 100)       # default ~4.8% annual
rf[300:480] = 0.35 / 100           # late 1980s–1990s
rf[480:600] = 0.15 / 100           # post-GFC low rate era
rf[600:660] = 0.05 / 100           # 2010s zero lower bound
rf[660:]    = 0.45 / 100           # 2022+ normalization

factors_df = pd.DataFrame(raw, index=dates, columns=['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA'])
factors_df['RF'] = rf

# Momentum factor (UMD): higher vol, crash risk, mildly neg correlated with market
mom_mean = 0.60 / 100
mom_std  = 4.80 / 100
# Momentum crashes (occasional large negative months)
mom_residual = np.random.normal(0, mom_std * np.sqrt(1 - 0.15**2), T)
# Add crash events (roughly 2 per decade)
crash_idx = np.random.choice(T, 12, replace=False)
mom_residual[crash_idx] -= np.abs(np.random.normal(0.06, 0.02, 12))
factors_df['UMD'] = mom_mean - 0.15 * raw[:, 0] / stds[0] * mom_std + mom_residual

factor_names = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']

# Factor summary statistics
stats = factors_df[factor_names].describe().T[['mean', 'std']].copy()
stats['ann_return_%']  = (stats['mean'] * 12 * 100).round(2)
stats['ann_vol_%']     = (stats['std'] * np.sqrt(12) * 100).round(2)
stats['ann_sharpe']    = (stats['ann_return_%'] / stats['ann_vol_%']).round(3)
print(f"\nFactor Statistics:\n{stats[['ann_return_%','ann_vol_%','ann_sharpe']].to_string()}")

# ============================================================
# 2. GENERATE 10 TEST PORTFOLIOS
# ============================================================
print("\nSTEP 2: Generating 10 Test Portfolios...")

PORTFOLIOS = {
    'Name':      ['Market', 'Value', 'SmallGrowth', 'Quality', 'Conservative',
                   'LargeBlend', 'SmallValue', 'Momentum', 'DefensiveQuality', 'Balanced'],
    'Mkt_Beta':  [ 0.95,  0.75,  1.10,  0.50,  0.60,  1.00,  0.85,  0.90,  0.40,  0.80],
    'SMB_Beta':  [ 0.05,  0.10,  0.85,  0.10,  0.10, -0.15,  0.80,  0.20,  0.05,  0.30],
    'HML_Beta':  [-0.10,  0.85, -0.60,  0.20,  0.40,  0.10,  0.75, -0.10,  0.30,  0.20],
    'RMW_Beta':  [ 0.10,  0.20, -0.20,  0.80,  0.15,  0.05,  0.15,  0.10,  0.70,  0.20],
    'CMA_Beta':  [ 0.05,  0.30, -0.30,  0.20,  0.75,  0.10,  0.25,  0.05,  0.25,  0.20],
    'UMD_Beta':  [ 0.05,  0.00, -0.10,  0.05,  0.00,  0.00,  0.00,  0.60,  0.05,  0.10],
    'Alpha':     [ 0.02,  0.03, -0.02,  0.03,  0.01,  0.00,  0.02,  0.04,  0.02,  0.01],  # monthly %
    'Noise_std': [ 1.50,  1.80,  2.00,  1.60,  1.20,  1.00,  1.70,  2.20,  1.40,  1.30],
}
params = pd.DataFrame(PORTFOLIOS).set_index('Name')

portfolio_rets = pd.DataFrame(index=dates)
for name, row in params.iterrows():
    portfolio_rets[name] = (
        row['Alpha'] / 100
        + row['Mkt_Beta'] * factors_df['Mkt_RF']
        + row['SMB_Beta'] * factors_df['SMB']
        + row['HML_Beta'] * factors_df['HML']
        + row['RMW_Beta'] * factors_df['RMW']
        + row['CMA_Beta'] * factors_df['CMA']
        + row['UMD_Beta'] * factors_df['UMD']
        + np.random.normal(0, row['Noise_std'] / 100, T)
    )

excess_rets = portfolio_rets.subtract(factors_df['RF'], axis=0)

ann_rets = (portfolio_rets.mean() * 12 * 100).round(2)
ann_vols = (portfolio_rets.std() * np.sqrt(12) * 100).round(2)
sharpes  = (ann_rets / ann_vols).round(3)
print(f"  {'Portfolio':<22} {'Ann.Ret%':>9} {'Ann.Vol%':>9} {'Sharpe':>8}")
for n in portfolio_rets.columns:
    print(f"  {n:<22} {ann_rets[n]:>9.2f} {ann_vols[n]:>9.2f} {sharpes[n]:>8.3f}")

# ============================================================
# 3. FACTOR VISUALIZATIONS
# ============================================================
print("\nSTEP 3: Factor Visualizations...")

# Cumulative factor returns
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, f in enumerate(factor_names):
    cum = (1 + factors_df[f]).cumprod()
    axes[i].plot(cum.index, cum.values, color=colors[i], lw=1.5)
    axes[i].axhline(1.0, color='grey', lw=0.8, ls='--')
    axes[i].set_title(f'{f}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Growth of $1')
    final_val = cum.iloc[-1]
    axes[i].annotate(f'${final_val:.1f}', xy=(cum.index[-1], final_val),
                      fontsize=10, fontweight='bold', color=colors[i],
                      xytext=(-50, 8), textcoords='offset points')
plt.suptitle('Cumulative Growth of $1 Invested in Each Factor  (1963–2023)',
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_cumulative_factor_returns.png', dpi=150, bbox_inches='tight')
plt.close()

# Factor correlation heatmap
fig, ax = plt.subplots(figsize=(8, 7))
factor_corr = factors_df[factor_names].corr()
mask = np.triu(np.ones_like(factor_corr, dtype=bool), k=1)
sns.heatmap(factor_corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={'size': 11})
ax.set_title('Factor Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/02_factor_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

# Rolling 12-month Sharpe ratios for each factor
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()
for i, f in enumerate(factor_names):
    roll_mean = factors_df[f].rolling(12).mean() * 12
    roll_std  = factors_df[f].rolling(12).std()  * np.sqrt(12)
    roll_sr   = (roll_mean / roll_std).dropna()
    axes[i].plot(roll_sr.index, roll_sr.values, color=colors[i], lw=1.2)
    axes[i].axhline(0, color='black', lw=0.8)
    axes[i].fill_between(roll_sr.index, 0, roll_sr.values,
                          where=(roll_sr.values > 0), alpha=0.2, color=colors[i])
    axes[i].fill_between(roll_sr.index, 0, roll_sr.values,
                          where=(roll_sr.values < 0), alpha=0.2, color='red')
    axes[i].set_title(f'{f} — Rolling 12M Sharpe', fontsize=11, fontweight='bold')
plt.suptitle('Rolling 12-Month Sharpe Ratios by Factor', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/03_rolling_sharpe_ratios.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. OLS REGRESSIONS: CAPM, FF3, FF5, FF5+MOM
# ============================================================
print("STEP 4: Running Asset Pricing Regressions...")

all_results = []
for port in excess_rets.columns:
    y = excess_rets[port].values
    row = {'Portfolio': port}

    # CAPM
    X = sm.add_constant(factors_df[['Mkt_RF']].values)
    r = sm.OLS(y, X).fit(cov_type='HC3')
    row.update({'CAPM_alpha': r.params[0]*100, 'CAPM_t': r.tvalues[0],
                 'CAPM_R2': r.rsquared, 'CAPM_beta': r.params[1]})

    # FF3
    X = sm.add_constant(factors_df[['Mkt_RF','SMB','HML']].values)
    r = sm.OLS(y, X).fit(cov_type='HC3')
    row.update({'FF3_alpha': r.params[0]*100, 'FF3_t': r.tvalues[0],
                 'FF3_R2': r.rsquared})

    # FF5
    X = sm.add_constant(factors_df[['Mkt_RF','SMB','HML','RMW','CMA']].values)
    r = sm.OLS(y, X).fit(cov_type='HC3')
    row.update({'FF5_alpha': r.params[0]*100, 'FF5_t': r.tvalues[0],
                 'FF5_R2': r.rsquared})

    # FF5 + Momentum
    X = sm.add_constant(factors_df[['Mkt_RF','SMB','HML','RMW','CMA','UMD']].values)
    r = sm.OLS(y, X).fit(cov_type='HC3')
    row.update({'FF5M_alpha': r.params[0]*100, 'FF5M_t': r.tvalues[0],
                 'FF5M_R2': r.rsquared})

    all_results.append(row)

reg_df = pd.DataFrame(all_results).set_index('Portfolio')
reg_df.to_csv('regression_results.csv')

print(f"\nR² by Portfolio:\n{reg_df[['CAPM_R2','FF3_R2','FF5_R2','FF5M_R2']].round(3).to_string()}")
print(f"\nMonthly Alpha (%):\n{reg_df[['CAPM_alpha','FF3_alpha','FF5_alpha','FF5M_alpha']].round(4).to_string()}")

# ============================================================
# 5. REGRESSION COMPARISON PLOTS
# ============================================================
print("\nSTEP 5: Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# R² grouped bar
r2_data = reg_df[['CAPM_R2','FF3_R2','FF5_R2','FF5M_R2']].copy()
r2_data.columns = ['CAPM','FF3','FF5','FF5+Mom']
r2_data.plot(kind='bar', ax=axes[0], color=colors[:4], alpha=0.85, edgecolor='white', width=0.75)
axes[0].set_title('Model R² by Portfolio', fontsize=13, fontweight='bold')
axes[0].set_ylabel('R²')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=35, labelsize=8)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].axhline(0.9, color='red', ls=':', lw=1, alpha=0.5, label='R²=0.90')

# Alpha grouped bar
alpha_data = reg_df[['CAPM_alpha','FF3_alpha','FF5_alpha','FF5M_alpha']].copy()
alpha_data.columns = ['CAPM','FF3','FF5','FF5+Mom']
alpha_data.plot(kind='bar', ax=axes[1], color=colors[:4], alpha=0.85, edgecolor='white', width=0.75)
axes[1].set_title('Monthly Alpha (%) by Portfolio', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Alpha (%/month)')
axes[1].tick_params(axis='x', rotation=35, labelsize=8)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/04_r2_alpha_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# R² improvement waterfall (average across portfolios)
avg_r2 = {
    'CAPM':     reg_df['CAPM_R2'].mean(),
    'FF3':      reg_df['FF3_R2'].mean(),
    'FF5':      reg_df['FF5_R2'].mean(),
    'FF5+Mom':  reg_df['FF5M_R2'].mean(),
}
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(avg_r2.keys(), avg_r2.values(), color=colors[:4], alpha=0.85,
               edgecolor='white', width=0.5)
for bar, val in zip(bars, avg_r2.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Average R² Across 10 Portfolios by Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Average R²')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/05_avg_r2_by_model.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. ROLLING FACTOR EXPOSURES
# ============================================================
print("STEP 6: Rolling Betas...")

WINDOW = 36
PORT   = 'Momentum'  # Momentum portfolio — interesting time-varying exposures

y_roll = excess_rets[PORT]
X_roll = sm.add_constant(factors_df[['Mkt_RF','SMB','HML','RMW','CMA','UMD']])
rols   = RollingOLS(y_roll, X_roll, window=WINDOW).fit()
rolling_params = rols.params.dropna()

param_labels = {
    'const': ('Alpha (monthly %)', lambda x: x*100),
    'Mkt_RF': ('Market Beta', lambda x: x),
    'SMB':    ('SMB Beta', lambda x: x),
    'HML':    ('HML Beta', lambda x: x),
    'RMW':    ('RMW Beta', lambda x: x),
    'CMA':    ('CMA Beta', lambda x: x),
    'UMD':    ('UMD (Momentum) Beta', lambda x: x),
}
true_vals = {
    'const':  params.loc[PORT, 'Alpha'] / 100 * 100,   # in %
    'Mkt_RF': params.loc[PORT, 'Mkt_Beta'],
    'SMB':    params.loc[PORT, 'SMB_Beta'],
    'HML':    params.loc[PORT, 'HML_Beta'],
    'RMW':    params.loc[PORT, 'RMW_Beta'],
    'CMA':    params.loc[PORT, 'CMA_Beta'],
    'UMD':    params.loc[PORT, 'UMD_Beta'],
}

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()
for i, (col, (label, transform)) in enumerate(param_labels.items()):
    vals = transform(rolling_params[col])
    axes[i].plot(vals.index, vals.values, color=colors[i], lw=1.5, label='Estimated')
    true = true_vals[col]
    axes[i].axhline(true, color='red', ls='--', lw=1.5, label=f'True = {true:.3f}')
    axes[i].set_title(f'Rolling {label}', fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=8)
axes[-1].set_visible(False)
plt.suptitle(f'Rolling 36-Month FF5+Mom Factor Exposures — {PORT} Portfolio',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/06_rolling_betas.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. RISK-RETURN SCATTER
# ============================================================
print("STEP 7: Risk-Return Plot...")

port_stats = pd.DataFrame({
    'Ann. Return (%)': ann_rets,
    'Ann. Vol (%)':    ann_vols,
    'Sharpe':          sharpes,
})

fig, ax = plt.subplots(figsize=(11, 7))
for i, port in enumerate(port_stats.index):
    ax.scatter(port_stats.loc[port, 'Ann. Vol (%)'],
               port_stats.loc[port, 'Ann. Return (%)'],
               s=140, color=colors[i % 10], zorder=5, edgecolors='white', linewidth=0.8)
    ax.annotate(port,
                 (port_stats.loc[port, 'Ann. Vol (%)'], port_stats.loc[port, 'Ann. Return (%)']),
                 textcoords='offset points', xytext=(7, 4), fontsize=9)

# Capital Market Line
rf_ann = factors_df['RF'].mean() * 12 * 100
mkt_ret = port_stats.loc['Market', 'Ann. Return (%)']
mkt_vol = port_stats.loc['Market', 'Ann. Vol (%)']
slope   = (mkt_ret - rf_ann) / mkt_vol
vols    = np.linspace(0, port_stats['Ann. Vol (%)'].max() * 1.15, 100)
ax.plot(vols, rf_ann + slope * vols, 'k--', lw=1.3, alpha=0.6, label='Capital Market Line')
ax.axhline(rf_ann, color='grey', ls=':', lw=1, label=f'Risk-Free ({rf_ann:.1f}% ann.)')

ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
ax.set_ylabel('Annualized Return (%)', fontsize=12)
ax.set_title('Risk–Return Tradeoff Across Portfolios', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plots/07_risk_return_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ All plots saved to plots/")
print(f"\nAverage R² by model:")
for k, v in avg_r2.items():
    print(f"  {k:<12}: {v:.4f}")
