"""
Chapter 2: regret and exploration of VAPE under different context structures.

Cases: isotropic baseline, single direction, fixed sparse support, correlated
i.i.d. contexts (sharpened confidence radius, Theorem 2.5) and the probabilistic
subspace setting (T = 8000).

Outputs: logdet_all.png, eigenvalues_grid.png, exploration_all.png,
regret_all.png, rate_check.png, corr_validation.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ── Global parameters ─────────────────────────────────────────────────────────
d = 10
Bx = 1.0; By = 2.0; Btheta = 1.0; lam = 1.0; noise_std = 0.3
N_SEEDS = 15; T_SHORT = 800; T_PROB = 8000
SEEDS = list(range(N_SEEDS))
s3 = 3

# ── Colors / names ────────────────────────────────────────────────────────────
COLORS = {
    'isotropic':    '#2166ac',
    'single_dir':   '#d6604d',
    'fixed_sparse': '#4dac26',
    'correlation':  '#e07b00',
    'prob_sub':     '#1a9850',
    'intermediate': '#b2182b',
}
NAMES = {
    'isotropic':    'Isotropic baseline',
    'single_dir':   'Single direction',
    'fixed_sparse': 'Fixed sparse (s=3)',
    'correlation':  r'Correlation i.i.d. (Thm\,2.5)',
    'prob_sub':     'Probabilistic subspace',
    'intermediate': r'Sequential / Prop. ($\ell=0.08$)',
}
SHORT_KEYS = ['isotropic', 'single_dir', 'fixed_sparse', 'correlation']

# ── Confidence radii ──────────────────────────────────────────────────────────
def C_rad_std(T, d_eff):
    """Standard confidence radius (uniform over R^d)."""
    return By * np.sqrt(d_eff * np.log(1 + Bx**2 * T / 1e-4)) + Btheta

def beta_sharp_val(T, delta=None):
    """Sharpened radius under i.i.d. context independence (Lemma 2.4)."""
    if delta is None:
        delta = T**(-4)
    return By * np.sqrt(2.0 * np.log(2.0 * T / delta)) + Btheta

def make_mu(eps, C):
    return eps / (C + 1e-12)

# ── mu selectors ──────────────────────────────────────────────────────────────
def mu_iso(T):
    return make_mu((d**2 * np.log(T)**2 / T)**(1/3), C_rad_std(T, d))

def mu_single(T):
    return make_mu((np.log(T)**2 / T)**(1/3), C_rad_std(T, 1))

def mu_fixed(T):
    return make_mu((s3**2 * np.log(T)**2 / T)**(1/3), C_rad_std(T, s3))

def mu_var(T):
    return make_mu((d**2 * np.log(T)**2 / T)**(1/3), C_rad_std(T, d))

def mu_corr_fn(T):
    """mu for correlation i.i.d. case: uses sharpened beta (no sqrt(d))."""
    eps = (d * np.log(max(T, 2))**2 / T)**(1/3)
    return make_mu(eps, beta_sharp_val(T))

# ── Context samplers ──────────────────────────────────────────────────────────
def iso_ctx(rng):
    x = rng.standard_normal(d); return x / np.linalg.norm(x) * Bx

def single_ctx(rng):
    x = np.zeros(d); x[0] = rng.uniform(-Bx, Bx); return x

def fixed_ctx(rng):
    x = np.zeros(d); v = rng.standard_normal(s3)
    x[:s3] = v / np.linalg.norm(v) * Bx; return x

def var_ctx(rng):
    x = np.zeros(d); sup = rng.choice(d, size=s3, replace=False)
    v = rng.standard_normal(s3); x[sup] = v / np.linalg.norm(v) * Bx; return x

# Correlation: i.i.d. isotropic contexts independent of noise (same distribution
# as baseline, but different mu from the sharpened confidence bound).
corr_ctx = iso_ctx

# Theta vectors
theta_iso    = np.ones(d) / np.sqrt(d)
theta_single = np.zeros(d); theta_single[0] = 1.0
theta_fixed  = np.zeros(d); theta_fixed[:s3] = 1/np.sqrt(s3)
theta_var    = np.zeros(d); theta_var[:s3]   = 1/np.sqrt(s3)
theta_corr   = theta_iso.copy()

# Probabilistic subspace
sP = 3; deltaP = 0.10
IP = list(range(sP)); IcP = list(range(sP, d))
thetaP = np.zeros(d); thetaP[:sP] = 1/np.sqrt(sP)
alpha_p = T_PROB**(-4); delta0_p = T_PROB**(-4)

def L_exact(T_, s_, d_, delta_, delta0_):
    azuma = np.sqrt(T_ * np.log(1/delta0_) / 2)
    bad   = delta_ * T_ + azuma
    return (s_ * np.log(1 + T_ * Bx**2 / s_) +
            (d_ - s_) * np.log(1 + Bx**2 / (d_ - s_) * bad))

L_val = L_exact(T_PROB, sP, d, deltaP, delta0_p)
C_prob = By * np.sqrt(2 * L_val / alpha_p) + Btheta
eps_prob = (L_val**2 / T_PROB)**(1/3)
mu_prob_val = eps_prob / (C_prob + 1e-12)

def prob_ctx(rng):
    x = np.zeros(d)
    if rng.random() > deltaP:
        v = rng.standard_normal(sP);       x[IP]  = v / np.linalg.norm(v) * Bx
    else:
        v = rng.standard_normal(d - sP);   x[IcP] = v / np.linalg.norm(v) * Bx
    return x, False  # (context, in_Ac) — Ac flag unused here

# ── Core VAPE simulation ───────────────────────────────────────────────────────
def run_case(name, T, sample_ctx, theta_star, mu, seeds,
             prob_mode=False, track_bound=False):
    """
    Run standard per-round gatekeeper VAPE.
    track_bound: record per-VA-round estimation error and confidence widths
                 (for the dedicated correlation validation figure).
    """
    all_logdet, all_eigs, all_cumG, all_cumR = [], [], [], []
    all_bd_err, all_bd_std, all_bd_sharp = [], [], []

    for seed in seeds:
        rng    = np.random.default_rng(seed)
        V      = lam * np.eye(d); V_inv = (1/lam) * np.eye(d)
        sum_xy = np.zeros(d)
        logdet_s = np.zeros(T); eigs_s = np.zeros((T, d))
        cumG_s   = np.zeros(T); cumR_s = np.zeros(T)
        bd_err = []; bd_std_l = []; bd_sharp_l = []
        G = 0; R = 0.0

        for t in range(T):
            x_t = prob_ctx(rng)[0] if prob_mode else sample_ctx(rng)
            theta_hat = V_inv @ sum_xy          # ridge estimate (before update)
            Vx   = V_inv @ x_t
            gate = np.sqrt(np.clip(x_t @ Vx, 0, None))
            ev   = np.linalg.eigvalsh(V)
            eigs_s[t]   = ev
            logdet_s[t] = np.sum(np.log(np.maximum(ev, 1e-12)))

            if gate > mu:                        # VA round
                p_t  = rng.uniform(-By, By)
                y_t  = x_t @ theta_star + rng.normal(0, noise_std)
                o_t  = 1.0 if p_t <= y_t else 0.0
                ytil = 2 * By * (o_t - 0.5)
                denom = 1.0 + x_t @ Vx
                V_inv -= np.outer(Vx, Vx) / denom
                V     += np.outer(x_t, x_t)
                sum_xy += ytil * x_t
                G += 1
                if track_bound:
                    err  = abs(x_t @ theta_hat - x_t @ theta_star)
                    b_s  = C_rad_std(T, d) * gate
                    b_sh = beta_sharp_val(T) * gate
                    bd_err.append(err)
                    bd_std_l.append(b_s)
                    bd_sharp_l.append(b_sh)
            else:                                # PE round
                p_t = x_t @ theta_hat

            r_t        = (p_t - x_t @ theta_star)**2 / (2 * By)
            cumG_s[t]  = G; R += r_t; cumR_s[t] = R

        all_logdet.append(logdet_s); all_eigs.append(eigs_s)
        all_cumG.append(cumG_s);     all_cumR.append(cumR_s)
        if track_bound:
            all_bd_err.append(bd_err)
            all_bd_std.append(bd_std_l)
            all_bd_sharp.append(bd_sharp_l)

    res = {'logdet': np.array(all_logdet), 'eigs': np.array(all_eigs),
           'cumG':   np.array(all_cumG),   'cumR': np.array(all_cumR)}
    if track_bound:
        res['bound_data'] = {
            'err':   all_bd_err,
            'std':   all_bd_std,
            'sharp': all_bd_sharp,
        }
    return res


def run_sequential(T, sample_ctx, theta_star, ell, seeds):
    """
    Sequential two-phase VAPE (Prop. A.5): pure VA until
    lambda_min(V_t) >= B_x^2 / mu_seq^2, then pure PE.
    With ell = 0.08 and T = 800, the threshold is never reached,
    so exploration is all-VA and regret is near-linear — illustrating
    that the approach is vacuous.
    """
    eps_seq  = (d / ell * np.log(max(T, 2)) / T)**(1/3)
    mu_seq   = make_mu(eps_seq, C_rad_std(T, d))
    threshold = Bx**2 / mu_seq**2

    all_cumG, all_cumR = [], []
    for seed in seeds:
        rng    = np.random.default_rng(seed)
        V      = lam * np.eye(d); V_inv = (1/lam) * np.eye(d)
        sum_xy = np.zeros(d)
        G = 0; R = 0.0
        cumG_s = np.zeros(T); cumR_s = np.zeros(T)
        in_va  = True

        for t in range(T):
            x_t = sample_ctx(rng)
            if in_va and np.linalg.eigvalsh(V)[0] >= threshold:
                in_va = False
            theta_hat = V_inv @ sum_xy

            if in_va:
                p_t  = rng.uniform(-By, By)
                y_t  = x_t @ theta_star + rng.normal(0, noise_std)
                o_t  = 1.0 if p_t <= y_t else 0.0
                ytil = 2 * By * (o_t - 0.5)
                Vx   = V_inv @ x_t
                denom = 1.0 + x_t @ Vx
                V_inv -= np.outer(Vx, Vx) / denom
                V     += np.outer(x_t, x_t)
                sum_xy += ytil * x_t
                G += 1
            else:
                p_t = x_t @ theta_hat

            r_t       = (p_t - x_t @ theta_star)**2 / (2 * By)
            cumG_s[t] = G; R += r_t; cumR_s[t] = R

        all_cumG.append(cumG_s); all_cumR.append(cumR_s)
    return np.array(all_cumG), np.array(all_cumR)


# ── Run all simulations ────────────────────────────────────────────────────────
T = T_SHORT
print("Running simulations (15 seeds × 4 cases + prob_sub) ...")
res_iso  = run_case('isotropic',    T, iso_ctx,    theta_iso,    mu_iso(T),      SEEDS)
res_sd   = run_case('single_dir',   T, single_ctx, theta_single, mu_single(T),   SEEDS)
res_fs   = run_case('fixed_sparse', T, fixed_ctx,  theta_fixed,  mu_fixed(T),    SEEDS)
res_corr = run_case('correlation',  T, corr_ctx,   theta_corr,   mu_corr_fn(T),  SEEDS,
                    track_bound=True)
res_prob = run_case('prob_sub', T_PROB, prob_ctx,  thetaP,       mu_prob_val,    SEEDS,
                    prob_mode=True)
print("  Done.")

RESULTS = {
    'isotropic':    (res_iso,  T_SHORT),
    'single_dir':   (res_sd,   T_SHORT),
    'fixed_sparse': (res_fs,   T_SHORT),
    'correlation':  (res_corr, T_SHORT),
    'prob_sub':     (res_prob, T_PROB),
}

# =============================================================================
# FIGURE 1: logdet_all
# =============================================================================
print("Plotting logdet_all.png ...")
fig, ax = plt.subplots(figsize=(12, 6))
t_ax = np.arange(1, T_SHORT + 1)
ax.plot(t_ax, d  * np.log(t_ax + 1), color='gray', linestyle='--',
        linewidth=1.0, label=r'$d\log t$')
ax.plot(t_ax, s3 * np.log(t_ax + 1), color='gray', linestyle=':',
        linewidth=1.0, label=r'$s\log t$, $s=3$')
for key in SHORT_KEYS:
    res, _ = RESULTS[key]
    m = res['logdet'].mean(0); s_ = res['logdet'].std(0)
    ax.plot(t_ax, m, color=COLORS[key], label=NAMES[key], linewidth=1.4)
    ax.fill_between(t_ax, m - s_, m + s_, alpha=0.15, color=COLORS[key])
ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel(r'$\log\det V_t$', fontsize=12)
ax.set_title(r'Log-determinant — all $T=800$ settings', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'logdet_all.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: logdet_all.png")

# =============================================================================
# FIGURE 2: eigenvalues_grid  (2 × 3)
# =============================================================================
print("Plotting eigenvalues_grid.png ...")
grid_keys = ['isotropic', 'single_dir', 'fixed_sparse',
             'correlation', 'prob_sub']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
axes[5].set_visible(False)   # hide the unused 6th panel

for idx, key in enumerate(grid_keys):
    ax = axes[idx]
    res, T_ = RESULTS[key]
    eigs_mean = res['eigs'].mean(0)          # (T_, d)
    t_ax = np.arange(1, T_ + 1)
    for j in range(d):
        moving = eigs_mean[-1, j] > lam + 0.05
        col = COLORS[key] if moving else '#aaaaaa'
        ax.semilogy(t_ax, eigs_mean[:, j], color=col, linewidth=0.9, alpha=0.75)
    final_ld = res['logdet'].mean(0)[-1]
    ax.set_title(f"{NAMES[key]}\nfinal log det={final_ld:.1f}", fontsize=10)
    ax.set_xlabel('Round $t$', fontsize=9)
    ax.set_ylabel('Eigenvalue', fontsize=9)
    ax.tick_params(labelsize=8)
    if key == 'correlation':
        # Annotate: improvement is in beta, not in eigenvalues
        beta_s  = beta_sharp_val(T_)
        beta_b  = C_rad_std(T_, d)
        ratio   = beta_s / beta_b
        ax.text(0.05, 0.97,
                (rf'$\beta_{{\rm sharp}}/\beta_{{\rm std}}\approx{ratio:.2f}$'
                 r' (no eigenvalue freezing;' '\n' r'improvement via $\beta$)'),
                transform=ax.transAxes, fontsize=8, va='top',
                color=COLORS['correlation'],
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.suptitle('Eigenvalue trajectories — all cases', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'eigenvalues_grid.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: eigenvalues_grid.png")

# =============================================================================
# FIGURE 3: exploration_all
# =============================================================================
print("Plotting exploration_all.png ...")
fig, ax = plt.subplots(figsize=(12, 6))
t_ax = np.arange(1, T_SHORT + 1)
for key in SHORT_KEYS:
    res, _ = RESULTS[key]
    m = res['cumG'].mean(0); s_ = res['cumG'].std(0)
    ax.plot(t_ax, m, color=COLORS[key], label=NAMES[key], linewidth=1.4)
    ax.fill_between(t_ax, m - s_, m + s_, alpha=0.15, color=COLORS[key])
# Inset: probabilistic subspace
ax_in = ax.inset_axes([0.04, 0.52, 0.36, 0.44])
t_p = np.arange(1, T_PROB + 1)
m_p = res_prob['cumG'].mean(0); s_p = res_prob['cumG'].std(0)
ax_in.plot(t_p, m_p, color=COLORS['prob_sub'], linewidth=1.2)
ax_in.fill_between(t_p, m_p - s_p, m_p + s_p, alpha=0.15, color=COLORS['prob_sub'])
ax_in.set_title('Prob. subspace ($T=8000$)', fontsize=9)
ax_in.tick_params(labelsize=8)
ax_in.set_xlabel('Round $t$', fontsize=8)
ax_in.set_ylabel('$|G_t|$', fontsize=8)
ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Cumulative explorations $|G_t|$', fontsize=12)
ax.set_title('Exploration count — all settings', fontsize=13)
ax.legend(fontsize=9, ncol=2, loc='lower right')
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'exploration_all.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: exploration_all.png")

# =============================================================================
# FIGURE 4: regret_all
# =============================================================================
print("Plotting regret_all.png ...")
fig, ax = plt.subplots(figsize=(12, 6))
t_ax = np.arange(1, T_SHORT + 1)
for key in SHORT_KEYS:
    res, _ = RESULTS[key]
    m = res['cumR'].mean(0); s_ = res['cumR'].std(0)
    ax.plot(t_ax, m, color=COLORS[key], label=NAMES[key], linewidth=1.4)
    ax.fill_between(t_ax, m - s_, m + s_, alpha=0.15, color=COLORS[key])
ax_in = ax.inset_axes([0.04, 0.52, 0.36, 0.44])
m_p = res_prob['cumR'].mean(0); s_p = res_prob['cumR'].std(0)
t_p = np.arange(1, T_PROB + 1)
ax_in.plot(t_p, m_p, color=COLORS['prob_sub'], linewidth=1.2)
ax_in.fill_between(t_p, m_p - s_p, m_p + s_p, alpha=0.15, color=COLORS['prob_sub'])
ax_in.set_title('Prob. subspace ($T=8000$)', fontsize=9)
ax_in.tick_params(labelsize=8)
ax_in.set_xlabel('Round $t$', fontsize=8)
ax_in.set_ylabel('$R_t$', fontsize=8)
ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Cumulative regret $R_t$', fontsize=12)
ax.set_title('Cumulative regret — all settings', fontsize=13)
ax.legend(fontsize=9, ncol=2, loc='lower right')
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'regret_all.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: regret_all.png")

# =============================================================================
# FIGURE 5: rate_check  (log-log R_T vs T, and R_T vs s)
# =============================================================================
print("Plotting rate_check.png (extra short simulations) ...")
Ts_rate = [200, 400, 800, 1600, 3200]
seeds5  = list(range(5))
R_iso = []; R_fs = []; R_corr = []

for T_ in Ts_rate:
    r = run_case('iso_r',   T_, iso_ctx,   theta_iso,   mu_iso(T_),     seeds5)
    R_iso.append(r['cumR'][:, -1].mean())
    r = run_case('fs_r',    T_, fixed_ctx, theta_fixed, mu_fixed(T_),   seeds5)
    R_fs.append(r['cumR'][:, -1].mean())
    r = run_case('corr_r',  T_, corr_ctx,  theta_corr,  mu_corr_fn(T_), seeds5)
    R_corr.append(r['cumR'][:, -1].mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
Ts_arr = np.array(Ts_rate, dtype=float)

def safe_slope(X, Y):
    return np.polyfit(np.log(np.maximum(X, 1)), np.log(np.maximum(Y, 1e-12)), 1)[0]

sl_iso  = safe_slope(Ts_arr, np.array(R_iso))
sl_fs   = safe_slope(Ts_arr, np.array(R_fs))
sl_corr = safe_slope(Ts_arr, np.array(R_corr))

ax1.loglog(Ts_arr, R_iso,  'o-', color=COLORS['isotropic'],
           label=f'Isotropic (slope={sl_iso:.2f})', linewidth=1.4)
ax1.loglog(Ts_arr, R_fs,   's-', color=COLORS['fixed_sparse'],
           label=f'Fixed sparse (slope={sl_fs:.2f})', linewidth=1.4)
ax1.loglog(Ts_arr, R_corr, 'D-', color=COLORS['correlation'],
           label=f'Correlation i.i.d. (slope={sl_corr:.2f})', linewidth=1.4)
ref = np.array(R_iso)[0] * (Ts_arr / Ts_arr[0])**(2/3)
ax1.loglog(Ts_arr, ref, color='gray', linestyle='--', label='slope 2/3 ref')
ax1.set_xlabel('$T$', fontsize=12); ax1.set_ylabel('$R_T$', fontsize=12)
ax1.set_title('Regret vs $T$ (log-log)', fontsize=12)
ax1.legend(fontsize=9); ax1.tick_params(labelsize=10)

# Right: R_T vs s for fixed sparse support
T_rate2 = 1600; s_vals = [1, 2, 3, 5, 7, 10]; R_s = []
for sv in s_vals:
    th_sv = np.zeros(d); th_sv[:sv] = 1/np.sqrt(sv)
    def make_ctx(sv_=sv):
        def ctx(rng):
            x = np.zeros(d); v = rng.standard_normal(sv_)
            x[:sv_] = v / np.linalg.norm(v) * Bx; return x
        return ctx
    mu_sv = make_mu((sv**2 * np.log(T_rate2)**2 / T_rate2)**(1/3),
                    C_rad_std(T_rate2, sv))
    r = run_case(f'fs_s{sv}', T_rate2, make_ctx(sv), th_sv, mu_sv, seeds5)
    R_s.append(r['cumR'][:, -1].mean())

s_arr = np.array(s_vals, dtype=float); R_s_a = np.array(R_s)
sl_s  = safe_slope(s_arr, R_s_a)
ax2.loglog(s_arr, R_s_a, 'D-', color=COLORS['fixed_sparse'],
           label=f'Fixed sparse (slope={sl_s:.2f})', linewidth=1.4)
ref2 = R_s_a[0] * (s_arr / s_arr[0])**(2/3)
ax2.loglog(s_arr, ref2, color='gray', linestyle='--', label='slope 2/3 ref')
ax2.set_xlabel('$s$', fontsize=12); ax2.set_ylabel('$R_T$', fontsize=12)
ax2.set_title(f'Regret vs $s$ at $T={T_rate2}$ (log-log)', fontsize=12)
ax2.legend(fontsize=9); ax2.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'rate_check.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: rate_check.png")

# =============================================================================
# FIGURE 6: corr_validation  (dedicated correlation figure — 3 panels)
# =============================================================================
print("Plotting corr_validation.png ...")

# --- Panel A: confidence bound tracking over VA rounds ---
# Use the stored bound_data from res_corr.
bd = res_corr['bound_data']
# Align per-seed sequences to the same length (min length across seeds)
min_va = min(len(x) for x in bd['err'])
if min_va == 0:
    min_va = 1
err_mat   = np.array([x[:min_va] for x in bd['err']])      # (N_SEEDS, min_va)
std_mat   = np.array([x[:min_va] for x in bd['std']])
sharp_mat = np.array([x[:min_va] for x in bd['sharp']])

va_rounds = np.arange(1, min_va + 1)
err_mean   = err_mat.mean(0);   err_sd   = err_mat.std(0)
std_mean   = std_mat.mean(0);   std_sd   = std_mat.std(0)
sharp_mean = sharp_mat.mean(0); sharp_sd = sharp_mat.std(0)

# --- Panel B: exploration counts (baseline / sequential / i.i.d.) ---
print("  Running sequential intermediate simulation ...")
cumG_seq, cumR_seq = run_sequential(T, corr_ctx, theta_corr, ell=0.08, seeds=SEEDS)
print("  Done.")
t_ax_s = np.arange(1, T + 1)

m_iso_G  = res_iso['cumG'].mean(0);  s_iso_G  = res_iso['cumG'].std(0)
m_corr_G = res_corr['cumG'].mean(0); s_corr_G = res_corr['cumG'].std(0)
m_seq_G  = cumG_seq.mean(0);         s_seq_G  = cumG_seq.std(0)

m_iso_R  = res_iso['cumR'].mean(0);  s_iso_R  = res_iso['cumR'].std(0)
m_corr_R = res_corr['cumR'].mean(0); s_corr_R = res_corr['cumR'].std(0)
m_seq_R  = cumR_seq.mean(0);         s_seq_R  = cumR_seq.std(0)

# --- Panel C: R_T vs d (log-log, showing d^{1/3} vs d^{2/3} separation) ---
print("  Running d-sweep simulations ...")
d_vals = [2, 3, 5, 8, 12, 18, 25]
T_d    = 800
seeds_d = list(range(8))
R_iso_d = []; R_corr_d = []
for dv in d_vals:
    th_v = np.ones(dv) / np.sqrt(dv)
    def ctx_iso_d(rng, dv_=dv):
        x = rng.standard_normal(dv_); return x / np.linalg.norm(x) * Bx
    mu_iso_d  = make_mu((dv**2 * np.log(T_d)**2 / T_d)**(1/3),
                        By * np.sqrt(dv * np.log(1 + Bx**2*T_d/1e-4)) + Btheta)
    mu_corr_d = make_mu((dv * np.log(T_d)**2 / T_d)**(1/3), beta_sharp_val(T_d))

    # Run with d=dv (need local d for run_case — use inline loop)
    R_iso_v = []; R_corr_v = []
    for seed in seeds_d:
        for (mu_v, which) in [(mu_iso_d, 'iso'), (mu_corr_d, 'corr')]:
            rng_v  = np.random.default_rng(seed)
            V_v    = lam * np.eye(dv); V_inv_v = (1/lam) * np.eye(dv)
            s_xy   = np.zeros(dv)
            R_v    = 0.0
            for t in range(T_d):
                x_v    = rng_v.standard_normal(dv); x_v /= np.linalg.norm(x_v); x_v *= Bx
                th_hat = V_inv_v @ s_xy
                Vx_v   = V_inv_v @ x_v
                gate_v = np.sqrt(np.clip(x_v @ Vx_v, 0, None))
                if gate_v > mu_v:
                    p_v  = rng_v.uniform(-By, By)
                    y_v  = x_v @ th_v + rng_v.normal(0, noise_std)
                    o_v  = 1.0 if p_v <= y_v else 0.0
                    yt_v = 2 * By * (o_v - 0.5)
                    dn_v = 1.0 + x_v @ Vx_v
                    V_inv_v -= np.outer(Vx_v, Vx_v) / dn_v
                    V_v     += np.outer(x_v, x_v)
                    s_xy    += yt_v * x_v
                else:
                    p_v = x_v @ th_hat
                R_v += (p_v - x_v @ th_v)**2 / (2 * By)
            if which == 'iso':
                R_iso_v.append(R_v)
            else:
                R_corr_v.append(R_v)
    R_iso_d.append(np.mean(R_iso_v))
    R_corr_d.append(np.mean(R_corr_v))
print("  Done.")

d_arr = np.array(d_vals, dtype=float)
R_iso_d_a  = np.maximum(np.array(R_iso_d),  1e-6)
R_corr_d_a = np.maximum(np.array(R_corr_d), 1e-6)
sl_iso_d   = safe_slope(d_arr, R_iso_d_a)
sl_corr_d  = safe_slope(d_arr, R_corr_d_a)

# --- Assemble figure ---
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: bound tightness
axA.plot(va_rounds, sharp_mean, color=COLORS['correlation'], linewidth=1.8,
         label=r'$\beta_{\rm sharp}(\delta)\|x_t\|_{V^{-1}}$ (sharpened bound)')
axA.fill_between(va_rounds, sharp_mean - sharp_sd, sharp_mean + sharp_sd,
                 alpha=0.2, color=COLORS['correlation'])
axA.plot(va_rounds, std_mean, color=COLORS['isotropic'], linewidth=1.8, linestyle='--',
         label=r'$\beta_{\rm std}\|x_t\|_{V^{-1}}$ (standard bound)')
axA.fill_between(va_rounds, std_mean - std_sd, std_mean + std_sd,
                 alpha=0.15, color=COLORS['isotropic'])
axA.plot(va_rounds, err_mean, color='#333333', linewidth=1.4, linestyle=':',
         label=r'$|\hat{g}(x_t)-g(x_t)|$ (empirical error)')
axA.fill_between(va_rounds, np.maximum(err_mean - err_sd, 0), err_mean + err_sd,
                 alpha=0.12, color='#333333')
# Ratio annotation
avg_ratio = (std_mean / np.maximum(sharp_mean, 1e-9)).mean()
axA.text(0.97, 0.97, rf'Mean ratio $\beta_{{\rm std}}/\beta_{{\rm sharp}}\approx{avg_ratio:.1f}$'
         f'\n(theory: $\\sqrt{{d}}={np.sqrt(d):.1f}$)',
         transform=axA.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round', fc='white', alpha=0.85))
axA.set_xlabel('VA round', fontsize=11)
axA.set_ylabel('Confidence width', fontsize=11)
axA.set_title('Confidence bound tightening\n(Lemma 2.4)', fontsize=11)
axA.legend(fontsize=8, loc='upper right')
axA.tick_params(labelsize=9)

# Panel B: exploration counts
axB.plot(t_ax_s, m_iso_G,  color=COLORS['isotropic'],    linewidth=1.6,
         label='Baseline (standard $\\beta$)')
axB.fill_between(t_ax_s, m_iso_G - s_iso_G, m_iso_G + s_iso_G,
                 alpha=0.15, color=COLORS['isotropic'])
axB.plot(t_ax_s, m_seq_G,  color=COLORS['intermediate'],  linewidth=1.6, linestyle='--',
         label=r'Sequential / Prop. ($\ell=0.08$)')
axB.fill_between(t_ax_s, m_seq_G - s_seq_G, m_seq_G + s_seq_G,
                 alpha=0.15, color=COLORS['intermediate'])
axB.plot(t_ax_s, m_corr_G, color=COLORS['correlation'],   linewidth=1.6, linestyle='-.',
         label=r'Correlation i.i.d. (sharpened $\beta$)')
axB.fill_between(t_ax_s, m_corr_G - s_corr_G, m_corr_G + s_corr_G,
                 alpha=0.15, color=COLORS['correlation'])
axB.set_xlabel('Round $t$', fontsize=11)
axB.set_ylabel('Cumulative VA rounds $|G_t|$', fontsize=11)
axB.set_title('Exploration length comparison', fontsize=11)
axB.legend(fontsize=8); axB.tick_params(labelsize=9)

# Panel C: R_T vs d (log-log)
axC.loglog(d_arr, R_iso_d_a,  'o-', color=COLORS['isotropic'],
           label=f'Baseline (slope={sl_iso_d:.2f})', linewidth=1.6)
axC.loglog(d_arr, R_corr_d_a, 'D-', color=COLORS['correlation'],
           label=f'Correlation i.i.d. (slope={sl_corr_d:.2f})', linewidth=1.6)
ref_23 = R_iso_d_a[0] * (d_arr / d_arr[0])**(2/3)
ref_13 = R_corr_d_a[0] * (d_arr / d_arr[0])**(1/3)
axC.loglog(d_arr, ref_23, color='gray', linestyle='--', linewidth=1.0,
           label=r'$d^{2/3}$ ref')
axC.loglog(d_arr, ref_13, color='gray', linestyle=':', linewidth=1.0,
           label=r'$d^{1/3}$ ref')
axC.set_xlabel('Dimension $d$', fontsize=11)
axC.set_ylabel('$R_T$', fontsize=11)
axC.set_title(f'Regret vs $d$, $T={T_d}$ (log-log)', fontsize=11)
axC.legend(fontsize=8); axC.tick_params(labelsize=9)

fig.suptitle('Correlation i.i.d. case — confidence bound and rate validation',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'corr_validation.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: corr_validation.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"All figures saved to: {OUT}")
print("=" * 60)
for f in ['logdet_all.png', 'eigenvalues_grid.png', 'exploration_all.png',
          'regret_all.png', 'rate_check.png', 'corr_validation.png']:
    status = "OK" if os.path.exists(os.path.join(OUT, f)) else "MISSING"
    print(f"  [{status}] {f}")
print("\nDone.")
