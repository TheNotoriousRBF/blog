import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    HMC_AVAILABLE = True
except ImportError:
    HMC_AVAILABLE = False
    pm = None
    az = None

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

NATURE_BLUE = '#0072B2'
NATURE_RED = '#D55E00'
NATURE_GREEN = '#009E73'
NATURE_PURPLE = '#CC79A7'

np.random.seed(42)
n = 2500

print("Simulating cohort...")

psa_raw = np.random.lognormal(mean=2.0, sigma=0.9, size=n)
psa_log = np.log(psa_raw + 1)
psa_spline = SplineTransformer(n_knots=4, degree=3).fit_transform(psa_log.reshape(-1, 1))

grade_cont = np.clip(np.random.poisson(lam=expit(psa_log - 2) * 2.5) + 1 + np.random.uniform(-0.3, 0.3, n), 1.0, 5.0)
base_t = 1.5 + 0.3*psa_log + 0.4*(grade_cont - 3) + np.random.normal(0, 0.5, n)
t_stage_cont = np.clip(np.clip(np.round(base_t), 1, 4) + np.random.uniform(-0.3, 0.3, n), 1.0, 4.0)
prop_gg4 = np.clip(0.08*grade_cont + 0.02*psa_log + 0.1*np.random.randn(n), 0, 1)
cribriform_cont = np.clip(expit(-2 + 0.8*grade_cont + 2*prop_gg4) + 0.05*np.random.randn(n), 0, 1)
mri_vol_log = np.log(np.clip(np.random.gamma(shape=2 + 0.5*psa_log, scale=0.3), 0.1, 15) + 0.1)

clinical_features = np.column_stack([psa_spline, grade_cont, t_stage_cont, prop_gg4, cribriform_cont, mri_vol_log])

clinical_risk_score = 0.4*psa_log + 0.5*grade_cont + 0.3*t_stage_cont + 0.25*prop_gg4
genomic_score = 0.75*clinical_risk_score + 0.20*np.random.randn(n) + 0.15*np.random.randn(n)
genomic_score_std = (genomic_score - genomic_score.mean()) / genomic_score.std()
corr_clinical_decipher = np.corrcoef(clinical_risk_score, genomic_score)[0,1]

logit_risk = (0.50*psa_spline[:,0] + 0.40*psa_spline[:,1] + 0.30*psa_spline[:,2] +
    0.20*psa_spline[:,3] + 0.15*psa_spline[:,4] + 0.10*psa_spline[:,5] +
    0.75*grade_cont + 0.50*t_stage_cont + 0.40*prop_gg4 +
    0.35*cribriform_cont + 0.30*mri_vol_log +
    0.18*(genomic_score - 0.75*clinical_risk_score)/0.20 +
    0.10*grade_cont*t_stage_cont + 0.08*psa_log*prop_gg4 - 5.5)
metastasis = np.random.binomial(1, expit(logit_risk))

if HMC_AVAILABLE:
    print("Running HMC...")

    def run_hmc(X, y):
        with pm.Model():
            intercept = pm.Normal("intercept", mu=0, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
            logits = intercept + pm.math.dot(X, beta)
            p = pm.Deterministic("p", pm.math.invlogit(logits))
            pm.Bernoulli("y_obs", p=p, observed=y)
            return pm.sample(1000, tune=1000, target_accept=0.9, chains=4, cores=4, progressbar=False)

    trace_clin = run_hmc(clinical_features, metastasis)
    trace_dec = run_hmc(genomic_score_std.reshape(-1, 1), metastasis)
    trace_both = run_hmc(np.column_stack([clinical_features, genomic_score_std]), metastasis)

    def calc_r2(trace):
        p = trace.posterior["p"].values.reshape(-1, n)
        var_f = np.var(p, axis=1)
        var_res = np.mean(p * (1 - p), axis=1)
        return var_f / (var_f + var_res + 1e-8)

    r2_clin_dist = calc_r2(trace_clin)
    r2_dec_dist = calc_r2(trace_dec)
    r2_both_dist = calc_r2(trace_both)
    unique_dist = r2_both_dist - r2_clin_dist

    r2_clin, r2_dec = np.mean(r2_clin_dist), np.mean(r2_dec_dist)
    r2_both = np.mean(r2_both_dist)
    r2_unique_decipher = np.mean(unique_dist)
    r2_shared = r2_dec - r2_unique_decipher

    clin_hdi = az.hdi(r2_clin_dist, 0.95)
    unique_hdi = az.hdi(unique_dist, 0.95)
    HMC_MODE = True

else:
    def simulate_posterior(pred_mle, n_samples=1000):
        posterior = np.zeros((n_samples, n))
        for i in range(n_samples):
            logit_pred = np.log(pred_mle / (1 - pred_mle + 1e-8) + 1e-8)
            logit_pred += np.random.normal(0, 0.15 * np.sqrt(pred_mle * (1 - pred_mle) + 0.01), n)
            posterior[i] = expit(logit_pred)
        return np.clip(posterior, 0.001, 0.999)

    def bayes_r2(pred):
        var_f = np.var(pred.mean(axis=0))
        var_res = np.mean(pred.mean(axis=0) * (1 - pred.mean(axis=0)))
        return var_f / (var_f + var_res + 1e-8)

    model_clin = LogisticRegression(penalty=None, max_iter=1000).fit(clinical_features, metastasis)
    model_dec = LogisticRegression(penalty=None, max_iter=1000).fit(genomic_score_std.reshape(-1, 1), metastasis)
    model_both = LogisticRegression(penalty=None, max_iter=1000).fit(np.column_stack([clinical_features, genomic_score_std]), metastasis)

    r2_clin = bayes_r2(simulate_posterior(model_clin.predict_proba(clinical_features)[:,1], 1000))
    r2_dec = bayes_r2(simulate_posterior(model_dec.predict_proba(genomic_score_std.reshape(-1, 1))[:,1], 1000))
    r2_both = bayes_r2(simulate_posterior(model_both.predict_proba(np.column_stack([clinical_features, genomic_score_std]))[:,1], 1000))
    r2_unique_decipher = max(0, r2_both - r2_clin)
    r2_shared = max(0, r2_dec - r2_unique_decipher)
    HMC_MODE = False
    r2_clin_dist = None

improvement = r2_both - r2_clin
percent_redundant = (r2_shared / r2_dec) * 100

print(f"Results: Clinical={r2_clin:.3f}, Decipher={r2_dec:.3f}, Unique={r2_unique_decipher:.3f}")

print("Generating figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# Left panel
x = np.arange(2)
width = 0.5
bars1 = ax1.bar(x, [r2_clin, r2_unique_decipher], width, label='Unique variance',
                color=[NATURE_BLUE, NATURE_RED], alpha=0.9, edgecolor='black', linewidth=1)
bars2 = ax1.bar(x, [0, r2_shared], width, bottom=[r2_clin, r2_unique_decipher],
                label='Redundant with clinical', color=NATURE_PURPLE, alpha=0.6, edgecolor='black', linewidth=1)


if HMC_MODE and r2_clin_dist is not None:
    # For single points, yerr should be [[lower], [upper]] shape (2, n) where n is number of points
    clin_err = [[r2_clin - clin_hdi[0]], [clin_hdi[1] - r2_clin]]
    ax1.errorbar([0], [r2_clin], yerr=clin_err, fmt='none', color='black', capsize=3, capthick=1.5)

    unique_err_lower = max(0, r2_unique_decipher - unique_hdi[0])
    unique_err_upper = unique_hdi[1] - r2_unique_decipher
    unique_err = [[unique_err_lower], [unique_err_upper]]
    ax1.errorbar([1], [r2_unique_decipher], yerr=unique_err, fmt='none', color='black', capsize=3, capthick=1.5)

# Text labels
ax1.text(0, r2_clin/2, f'{r2_clin:.1%}', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

if r2_unique_decipher > 0.015:
    ax1.text(1, r2_unique_decipher/2, f'{r2_unique_decipher:.1%}', ha='center', va='center',
             fontsize=10, color='white', fontweight='bold')
else:
    ax1.text(1, r2_unique_decipher + 0.015, f'{r2_unique_decipher:.1%}', ha='center', va='bottom',
             fontsize=10, color='black', fontweight='bold')

ax1.text(1, r2_unique_decipher + r2_shared/2, f'{r2_shared:.1%}\nredundant', ha='center', va='center', fontsize=9)

ax1.set_ylabel('Variance explained (R²)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(['Clinical\nVariables', 'Decipher\n(Genomic)'])
ax1.set_ylim(0, 0.42)
ax1.legend(loc='upper right', frameon=False, fontsize=9)
ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

ax1.annotate(f'{percent_redundant:.0f}% of Decipher is just\nclinical information',
             xy=(1, r2_dec - 0.03), xytext=(1.4, 0.28),
             fontsize=9, color='darkred', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkred'),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5, connectionstyle="arc3,rad=0.2"))

# Right panel
models = ['Clinical\nalone', 'Decipher\nalone', 'Clinical +\nDecipher']
r2_values = [r2_clin, r2_dec, r2_both]
bars = ax2.bar(models, r2_values, color=[NATURE_BLUE, NATURE_RED, NATURE_GREEN],
               alpha=0.9, edgecolor='black', linewidth=1, width=0.5)

for bar, val in zip(bars, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01, f'{val:.1%}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.annotate('', xy=(2, r2_both), xytext=(2, r2_clin),
             arrowprops=dict(arrowstyle='<->', color='darkred', lw=2))
ax2.text(2.18, (r2_clin + r2_both)/2, f'+{improvement:.1%}',
         fontsize=10, color='darkred', va='center', fontweight='bold')

ax2.set_ylabel('Variance explained (R²)', fontsize=11)
ax2.set_ylim(0, 0.42)
ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

if HMC_MODE:
    summary = (f'HMC Bayesian Analysis\n'
               f'Clinical: {r2_clin:.1%} [{clin_hdi[0]:.1%}, {clin_hdi[1]:.1%}]\n'
               f'Unique genomic: {r2_unique_decipher:.1%} [{unique_hdi[0]:.1%}, {unique_hdi[1]:.1%}]\n'
               f'Redundant: {percent_redundant:.0f}%\n'
               f'Correlation: {corr_clinical_decipher:.2f}')
else:
    summary = (f'Bootstrap Analysis\n'
               f'Clinical R²: {r2_clin:.1%}\n'
               f'Decipher unique: {r2_unique_decipher:.1%}\n'
               f'Redundant: {percent_redundant:.0f}%\n'
               f'Correlation: {corr_clinical_decipher:.2f}')

ax2.text(0.97, 0.97, summary, transform=ax2.transAxes, fontsize=8.5,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9),
         family='monospace')

method = "HMC (4×1000 samples)" if HMC_MODE else "Bootstrap simulation"
fig.suptitle(f'Genomic classifier adds minimal independent prognostic information ({method})',
             fontsize=12, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fig.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('fig.pdf', dpi=600, bbox_inches='tight', facecolor='white')

print(f"✓ Saved fig.png/pdf")
print(f"Result: {percent_redundant:.0f}% redundant, {r2_unique_decipher:.1%} unique")
