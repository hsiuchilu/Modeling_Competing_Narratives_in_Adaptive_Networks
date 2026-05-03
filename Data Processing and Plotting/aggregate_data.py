import numpy as np
import pandas as pd
import os
import warnings
from scipy.stats import kurtosis, skew
from scipy.optimize import curve_fit

# Suppress warnings from curve_fit to keep console output clean
warnings.filterwarnings('ignore')

# Ensure the output directory exists
os.makedirs('data', exist_ok=True)

# ==========================================
# Common Helper Functions
# ==========================================

def generlized_logit_regression(x, k, b, Q):
    """Generalized logit regression function."""
    return k / ((1 + Q * np.exp(-b * x))**1/1)

def fitting(x, y):
    """Fit data to the generalized logit regression model."""
    try:
        params, _ = curve_fit(generlized_logit_regression, x, y, maxfev=1000, bounds=([-np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf]))
        k_fit, b_fit, q_fit = params
        x_axis = np.arange(1000)
        y_fit = generlized_logit_regression(x_axis, k_fit, b_fit, q_fit)
        y_test = y_fit[-1]
        
        # Filter invalid fits
        if k_fit > 1 or k_fit < 0.1 or q_fit < 0 or y_test < 0.2:
            k_fit, b_fit, q_fit, y_test = float('nan'), float('nan'), float('nan'), float('nan')
    except:
        y_fit, k_fit, b_fit, q_fit, y_test = [0, 0], float('nan'), float('nan'), float('nan'), float('nan')
    return y_fit, k_fit, q_fit, y_test, b_fit

def heat_data(Share, w, beta):
    """Format data for heatmap DataFrames."""
    return pd.DataFrame({'Value': Share, 'w': w, 'beta': beta}, index=[0])

def signle_and_both2(S, Mis):
    """Analyze single and both states for tipping points."""
    Single = []
    Tip = []
    Y = []
    for i in range(S.shape[0]):
        y1 = np.array(S)[i, :][np.array(Mis)[i, :] == 1]
        x1 = np.where(np.array(Mis)[i, :] == 1)[0]
        y0 = np.array(S)[i, :][np.array(Mis)[i, :] == 0]
        x0 = np.where(np.array(Mis)[i, :] == 0)[0]
        
        y_fit1, _, _, y_test1, _ = fitting(x1, y1)
        y_fit0, _, _, y_test0, _ = fitting(x0, y0)

        single1, single2 = False, False
        if np.max(y_fit1) > 0.25:
            single1 = True
            Tip.append(np.where(y_fit1 > 0.25)[0][0])
            Y.append(y_test1)
        if np.max(y_fit0) > 0.25:
            single2 = True
            Tip.append(np.where(y_fit0 > 0.25)[0][0])
            Y.append(y_test0)
            
        Single.append(single1)
        Single.append(single2)
        
    if len(Tip) == 0: Tip.append(np.nan)
    if len(Y) == 0: Y.append(np.nan)
    return Single, Tip, Y

def confid(s3, m3):
    """Calculate confidence intervals for dynamic models (Figure 5)."""
    Y0 = np.zeros(s3.shape)
    Y1 = np.zeros(s3.shape)
    x_axis = np.arange(1000)
    for i in range(np.array(s3).shape[0]):
        x_0 = np.where(np.array(m3)[i, :] == 0)[0]
        x_1 = np.where(np.array(m3)[i, :] == 1)[0]
        y_0 = np.array(s3)[i, :][x_0]
        y_1 = np.array(s3)[i, :][x_1]
        
        params0, _ = curve_fit(generlized_logit_regression, x_0, y_0, maxfev=1000, bounds=([-np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf]))
        Y0[i, :] = generlized_logit_regression(x_axis, *params0)
        
        params1, _ = curve_fit(generlized_logit_regression, x_1, y_1, maxfev=1000, bounds=([-np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf]))
        Y1[i, :] = generlized_logit_regression(x_axis, *params1)
    return np.concatenate((Y0, Y1))


# ==========================================
# 1. Figure 3 Data Extraction
# ==========================================
print("Processing Figure 1 data...")
w_fig1 = 0.4
alpha_fig1 = 0.5
share_path = f'degree/share_time_N_1000_w_{w_fig1}_alpha_{alpha_fig1}.npy'
misinfo_path = f'degree/mis_info_N_1000_w_{w_fig1}_alpha_{alpha_fig1}.npy'

pd.DataFrame(np.load(share_path)).to_csv(f'data/share_time_w{w_fig1}_alpha{alpha_fig1}.csv', index=False)
pd.DataFrame(np.load(misinfo_path)).to_csv(f'data/mis_info_w{w_fig1}_alpha{alpha_fig1}.csv', index=False)


# ==========================================
# 2. Complex Heatmaps Data Aggregation
# ==========================================
print("Processing Complex Heatmaps data... (This might take a while due to 41x41 loop)")
DF_h1, DF_h2, DF_h3, DF_h4, DF_h5, DF_h6 = [], [], [], [], [], []
alphas_heat = np.round(np.linspace(0, 1, 41), 3)
ws_heat = np.round(np.linspace(0, 1, 41), 3)

for alpha in alphas_heat:
    for w in ws_heat:
        try:
            # Load and concatenate arrays
            share = np.concatenate((np.load(f'alpha_w參數/share_time_zealot10_alhpa{alpha}_w{w}.npy'), 
                                    np.load(f'degree/share_time_w_{w}_alpha_{alpha}.npy')), axis=0)
            mis = np.concatenate((np.load(f'alpha_w參數/mis_info_zealot10_alhpa{alpha}_w{w}.npy'), 
                                  np.load(f'degree/mis_info_w_{w}_alpha_{alpha}.npy')), axis=0)
            mod = np.concatenate((np.load(f'alpha_w參數/Modularity_zealot10_alhpa{alpha}_w{w}.npy'), 
                                  np.load(f'degree/Modularity_w_{w}_alpha_{alpha}.npy')), axis=0)
            ass = np.concatenate((np.load(f'alpha_w參數/Assortativity_zealot10_alhpa{alpha}_w{w}.npy'), 
                                  np.load(f'degree/Assortativity_w_{w}_alpha_{alpha}.npy')), axis=0)
            
            S = share / 500
            x_axis = np.arange(1000)
            y = np.mean(S, axis=0)
            
            _, _, q_fit, _, _ = fitting(x_axis, y)
            single, tip, Y = signle_and_both2(S, mis)
            
            DF_h1.append(heat_data(Share=q_fit, w=w, beta=alpha))
            DF_h2.append(heat_data(Share=sum(single)/len(single), w=w, beta=alpha))
            DF_h3.append(heat_data(Share=np.nanmean(Y), w=w, beta=alpha))
            DF_h4.append(heat_data(Share=np.mean(mod), w=w, beta=alpha))
            DF_h5.append(heat_data(Share=np.mean(np.nan_to_num(ass, nan=1)), w=w, beta=alpha))
            DF_h6.append(heat_data(Share=np.mean(tip), w=w, beta=alpha))
            
        except FileNotFoundError:
            # Skip if file does not exist during testing
            continue

if DF_h1:
    pd.concat(DF_h1).reset_index(drop=True).pivot(index='w', columns='beta', values='Value').to_csv('data/heatmap_time_to_growth.csv')
    pd.concat(DF_h2).reset_index(drop=True).pivot(index='w', columns='beta', values='Value').to_csv('data/heatmap_prob_tau.csv')
    pd.concat(DF_h3).reset_index(drop=True).pivot(index='w', columns='beta', values='Value').to_csv('data/heatmap_f_1000.csv')
    pd.concat(DF_h4).reset_index(drop=True).pivot(index='w', columns='beta', values='Value').to_csv('data/heatmap_modularity.csv')
    pd.concat(DF_h5).reset_index(drop=True).pivot(index='w', columns='beta', values='Value').to_csv('data/heatmap_assortativity.csv')
    pd.concat(DF_h6).reset_index(drop=True).pivot(index='w', columns='beta', values='Value').to_csv('data/heatmap_f_inv_025.csv')


# ==========================================
# 3. Figure 4 Data Aggregation
# ==========================================
print("Processing Figure 4 data...")
W = np.round(np.linspace(0, 1, 21), 3)
Alpha = np.round(np.linspace(0, 1, 21), 3)
DF4_1, DF4_2, DF4_3 = [], [], []

for w in W:
    for alpha in Alpha:
        try:
            degrees = np.concatenate((np.load(f'degree/Degree_w_{w}_alpha_{alpha}.npy'), 
                                      np.load(f'degree/Degree2_w_{w}_alpha_{alpha}.npy')), axis=0)
            z = np.concatenate((np.load(f'degree/Zealot_w_{w}_alpha_{alpha}.npy').reshape(30, 10), 
                                np.load(f'degree/Zealot2_w_{w}_alpha_{alpha}.npy').reshape(20, 10)), axis=0)

            z_mean = [np.mean(degrees[j][z[j, :]]) for j in range(len(degrees))]
            
            DF4_1.append({'w': w, 'alpha': alpha, 'value': np.mean(z_mean)})
            DF4_2.append({'w': w, 'alpha': alpha, 'value': np.var(degrees)})
            DF4_3.append({'w': w, 'alpha': alpha, 'value': skew(np.mean(np.sort(degrees, axis=1), axis=0))})
        except FileNotFoundError:
            continue

if DF4_1:
    pd.DataFrame(DF4_1).pivot(index='w', columns='alpha', values='value').to_csv('data/fig4_zealot_degree.csv')
    pd.DataFrame(DF4_2).pivot(index='w', columns='alpha', values='value').to_csv('data/fig4_degree_var.csv')
    pd.DataFrame(DF4_3).pivot(index='w', columns='alpha', values='value').to_csv('data/fig4_degree_skew.csv')


# ==========================================
# 4. Figure 5 Data Aggregation
# ==========================================
print("Processing Figure 5 data...")
fig5_data = []
dynamics = ['constant2', 'linear2', 'convex2', 'concave2']

for w in [0.2, 0.5, 0.9]:
    for alpha in [0.2, 0.5, 0.9]:
        for dyn in dynamics:
            try:
                s = np.load(f'dynamic b1/share_time_{dyn}_w_{w}_alpha_{alpha}.npy') / 500
                m = np.load(f'dynamic b1/mis_info_{dyn}_w_{w}_alpha_{alpha}.npy')
                
                Y_fits = confid(s, m)
                mean_values = np.mean(Y_fits, axis=0)
                ci = 1.96 * (np.std(Y_fits, axis=0) / np.sqrt(Y_fits.shape[0]))
                
                for t in range(1000):
                    fig5_data.append({'w': w, 'alpha': alpha, 'dynamic': dyn, 'time': t, 'mean': mean_values[t], 'ci': ci[t]})
            except FileNotFoundError:
                continue

if fig5_data:
    pd.DataFrame(fig5_data).to_csv('data/fig5_trajectories.csv', index=False)


# ==========================================
# 5. Figure 6 Data Aggregation
# ==========================================
print("Processing Figure 6 data...")
fig6_traj = []
fig6_assort = []
alpha_fig6 = 0.5

for n_zealot_0 in [5, 10, 15, 20, 25]:
    try:
        zealot_ratio = n_zealot_0 / 5
        s = np.load(f'zealot2/share_time_zealot0_{n_zealot_0}_alpha_{alpha_fig6}.npy') / 500
        m = np.load(f'zealot2/mis_info_zealot0_{n_zealot_0}_alpha_{alpha_fig6}.npy')
        
        s_fit_0, s_fit_1 = np.zeros((s.shape[0], 1000)), np.zeros((s.shape[0], 1000))
        
        for i in range(s.shape[0]):
            x_0 = np.where(m[i, :] == 0)[0]
            x_1 = np.where(m[i, :] == 1)[0]
            
            params0, _ = curve_fit(generlized_logit_regression, x_0, s[i, :][x_0], maxfev=1000, bounds=([-np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf]))
            s_fit_0[i, :] = generlized_logit_regression(np.arange(1000), *params0)
            
            params1, _ = curve_fit(generlized_logit_regression, x_1, s[i, :][x_1], maxfev=1000, bounds=([-np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf]))
            s_fit_1[i, :] = generlized_logit_regression(np.arange(1000), *params1)
            
        y0_mean, y1_mean = np.mean(s_fit_0, axis=0), np.mean(s_fit_1, axis=0)
        
        for t in range(1000):
            fig6_traj.append({'time': t, 'value': y0_mean[t], 'zealot': zealot_ratio, 'state': 0})
            fig6_traj.append({'time': t, 'value': y1_mean[t], 'zealot': zealot_ratio, 'state': 1})
            
        a0 = np.load(f'zealot2/Assortativity0_zealot0_{n_zealot_0}_alpha_{alpha_fig6}.npy')
        a1 = np.load(f'zealot2/Assortativity1_zealot0_{n_zealot_0}_alpha_{alpha_fig6}.npy')
        fig6_assort.append({'zealot': zealot_ratio, 'A0': np.mean(a0), 'A1': np.mean(a1)})
    except FileNotFoundError:
        continue

if fig6_traj:
    pd.DataFrame(fig6_traj).to_csv('data/fig6_trajectories.csv', index=False)
    pd.DataFrame(fig6_assort).to_csv('data/fig6_assortativity.csv', index=False)

print("\nAll data aggregation is complete! You can now use Jupyter Notebook to generate figures.")
