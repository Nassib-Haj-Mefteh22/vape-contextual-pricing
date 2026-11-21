"""
Chapter 2: confidence ellipsoids and supporting diagnostics.

Main outputs: ellipsoid_single_dir.png, ellipsoid_fixed_sparse.png,
ellipsoid_var_sparse.png, block_epl.png, eta_necessary.png,
lambda_min_global.png. It also redraws earlier versions of the regret and
log-det figures; the versions shown in the report come from
ch2_context_structures.py.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ── Global simulation parameters ─────────────────────────────────────────────
d          = 10
Bx         = 1.0
By         = 2.0
Btheta     = 1.0
lam        = 1.0
noise_std  = 0.3
N_SEEDS    = 15
T_SHORT    = 800
T_PROB     = 8000
SEEDS      = list(range(N_SEEDS))

# ── Colors and display names (no internal labels) ────────────────────────────
COLORS = {
    'isotropic':    '#2166ac',
    'single_dir':   '#d6604d',
    'fixed_sparse': '#4dac26',
    'var_sparse':   '#8e6cb3',
    'global_cov':   '#f4a582',
    'block_leakage':'#92c5de',
    'block_abis':   '#b2182b',
    'prob_sub':     '#1a9850',
}
NAMES = {
    'isotropic':    'Isotropic baseline',
    'single_dir':   'Single direction',
    'fixed_sparse': 'Fixed sparse (s=3)',
    'var_sparse':   'Variable sparse (s=3)',
    'global_cov':   r'Global cov lb ($\ell$=0.5)',
    'block_leakage': r'Block leakage ($\eta$=0.1)',
    'block_abis':   r'Block-diagonal, no $\eta$',
    'prob_sub':     'Probabilistic subspace',
}
SHORT_KEYS = ['isotropic', 'single_dir', 'fixed_sparse', 'var_sparse',
              'global_cov', 'block_leakage', 'block_abis']

# ── Helper functions ──────────────────────────────────────────────────────────
def make_mu(T, eps, C_radius):
    return eps / (C_radius + 1e-10)

def C_rad(T, d_eff):
    return By * np.sqrt(d_eff * np.log(1 + Bx**2 * T / 1e-4)) + Btheta

# ── Core simulation (multi-seed) ──────────────────────────────────────────────
def run_case(name, T, sample_ctx, theta_star, mu, seeds, extra_tracking=None):
    all_logdet = []; all_eigs = []; all_cumG = []; all_cumR = []; all_extra = []
    for seed in seeds:
        rng      = np.random.default_rng(seed)
        V        = lam * np.eye(d)
        V_inv    = (1/lam) * np.eye(d)
        sum_xy   = np.zeros(d)
        logdet_s = np.zeros(T); eigs_s = np.zeros((T, d))
        cumG_s   = np.zeros(T); cumR_s = np.zeros(T)
        G = 0; R = 0.0
        if extra_tracking == 'block_eig':
            s_blk = 3
            I_blk = list(range(s_blk)); Ic_blk = list(range(s_blk, d))
            eig_I_s = np.zeros(T); eig_Ic_s = np.zeros(T)
        elif extra_tracking == 'prob_Ac':
            cumG_Ac_s = np.zeros(T); G_Ac = 0
        for t in range(T):
            if extra_tracking == 'prob_Ac':
                x_t, in_Ac = sample_ctx(rng)
            else:
                x_t = sample_ctx(rng)
            theta_hat = np.linalg.solve(V, sum_xy) if np.any(sum_xy != 0) else np.zeros(d)
            Vx   = V_inv @ x_t
            gate = np.sqrt(np.clip(x_t @ Vx, 0, None))
            ev   = np.linalg.eigvalsh(V)
            eigs_s[t]   = ev
            logdet_s[t] = np.sum(np.log(np.maximum(ev, 1e-12)))
            if extra_tracking == 'block_eig':
                ix  = np.ix_(I_blk,  I_blk)
                ix2 = np.ix_(Ic_blk, Ic_blk)
                eig_I_s[t]  = x_t[I_blk]  @ V_inv[ix]  @ x_t[I_blk]
                eig_Ic_s[t] = x_t[Ic_blk] @ V_inv[ix2] @ x_t[Ic_blk]
            if gate > mu:
                p_t    = rng.uniform(-By, By)
                y_t    = x_t @ theta_star + rng.normal(0, noise_std)
                o_t    = 1.0 if p_t <= y_t else 0.0
                ytil   = 2 * By * (o_t - 0.5)
                denom  = 1.0 + x_t @ Vx
                V_inv -= np.outer(Vx, Vx) / denom
                V     += np.outer(x_t, x_t)
                sum_xy += ytil * x_t
                G += 1
                if extra_tracking == 'prob_Ac' and in_Ac:
                    G_Ac += 1
            else:
                p_t = x_t @ theta_hat
            r_t      = (p_t - x_t @ theta_star)**2 / (2 * By)
            cumG_s[t] = G; R += r_t; cumR_s[t] = R
            if extra_tracking == 'prob_Ac':
                cumG_Ac_s[t] = G_Ac
        all_logdet.append(logdet_s); all_eigs.append(eigs_s)
        all_cumG.append(cumG_s);     all_cumR.append(cumR_s)
        if extra_tracking == 'block_eig':
            all_extra.append({'eig_I': eig_I_s, 'eig_Ic': eig_Ic_s})
        elif extra_tracking == 'prob_Ac':
            all_extra.append({'cumG_Ac': cumG_Ac_s})
    result = {'logdet': np.array(all_logdet), 'eigs': np.array(all_eigs),
              'cumG':   np.array(all_cumG),   'cumR': np.array(all_cumR)}
    if all_extra:
        result['extra'] = all_extra
    return result

# ── Case definitions ──────────────────────────────────────────────────────────
s3   = 3
ell4 = 0.5

def case1_ctx(rng):
    x = rng.standard_normal(d); return x / np.linalg.norm(x) * Bx
theta1 = np.ones(d) / np.sqrt(d)
def mu1(T): return make_mu(T, (d**2 * np.log(T)**2 / T)**(1/3), C_rad(T, d))

def case2_ctx(rng):
    x = np.zeros(d); x[0] = rng.uniform(-Bx, Bx); return x
theta2 = np.zeros(d); theta2[0] = 1.0
def mu2(T): return make_mu(T, (np.log(T)**2 / T)**(1/3), C_rad(T, 1))

theta3a = np.zeros(d); theta3a[:s3] = 1 / np.sqrt(s3)
def case3a_ctx(rng):
    x = np.zeros(d); v = rng.standard_normal(s3)
    x[:s3] = v / np.linalg.norm(v) * Bx; return x
def mu3a(T): return make_mu(T, (s3**2 * np.log(T)**2 / T)**(1/3), C_rad(T, s3))

theta3b = np.zeros(d); theta3b[:s3] = 1 / np.sqrt(s3)
def case3b_ctx(rng):
    x = np.zeros(d); sup = rng.choice(d, size=s3, replace=False)
    v = rng.standard_normal(s3); x[sup] = v / np.linalg.norm(v) * Bx; return x
def mu3b(T): return make_mu(T, (d**2 * np.log(T)**2 / T)**(1/3), C_rad(T, d))

rng4 = np.random.default_rng(0)
ev4  = ell4 + rng4.uniform(0, Bx**2 - ell4, d); L4 = np.diag(np.sqrt(ev4))
theta4 = np.zeros(d); theta4[0] = 1.0
def case4_ctx(rng):
    x = L4 @ rng.standard_normal(d); return x / np.linalg.norm(x) * Bx
def mu4(T): return make_mu(T, (d**2 * np.log(T)**2 / T)**(1/3), C_rad(T, d))

sA = 3; ellA = 0.5; etaA = 0.1
IA = np.arange(sA); IcA = np.arange(sA, d)
thetaA = np.zeros(d); thetaA[:sA] = 1 / np.sqrt(sA)
rngA   = np.random.default_rng(10)
AmA    = rngA.standard_normal((sA, sA))
SigA   = AmA @ AmA.T / sA + ellA * np.eye(sA)
LA     = np.linalg.cholesky(SigA)
def caseA_ctx(rng):
    x = np.zeros(d)
    xi = LA @ rng.standard_normal(sA); ni = np.linalg.norm(xi)
    if ni > 0: x[IA]  = xi / ni * np.sqrt(Bx**2 - etaA**2)
    xic = rng.standard_normal(d - sA); nic = np.linalg.norm(xic)
    if nic > 0: x[IcA] = xic / nic * etaA
    return x
def muA(T): return make_mu(T, (d**2 * np.log(T)**2 / T)**(1/3), C_rad(T, d))

sAb = 3; IAb = np.arange(sAb); IcAb = np.arange(sAb, d)
thetaAb = np.zeros(d); thetaAb[:sAb] = 1 / np.sqrt(sAb)
rngAb   = np.random.default_rng(20)
AmAb    = rngAb.standard_normal((sAb, sAb))
SigAb   = AmAb @ AmAb.T / sAb + 0.5 * np.eye(sAb)
LAb     = np.linalg.cholesky(SigAb)
def caseAb_ctx(rng):
    x = np.zeros(d); tgt = np.sqrt(Bx**2 / 2)
    xi = LAb @ rng.standard_normal(sAb); ni = np.linalg.norm(xi)
    if ni > 0: x[IAb]  = xi / ni * tgt
    xic = rng.standard_normal(d - sAb); nic = np.linalg.norm(xic)
    if nic > 0: x[IcAb] = xic / nic * tgt
    return x
def muAb(T): return make_mu(T, (d**2 * np.log(T)**2 / T)**(1/3), C_rad(T, d))

sP = 3; deltaP = 0.10
IP = list(range(sP)); IcP = list(range(sP, d))
thetaP = np.zeros(d); thetaP[:sP] = 1 / np.sqrt(sP)
alpha_prob  = T_PROB**(-4); delta0_prob = T_PROB**(-4)

def L_exact(T_, s_, d_, delta_, delta0_):
    bernstein_fluct = np.sqrt(T_ * np.log(1 / delta0_) / 2)
    bad   = delta_ * T_ + bernstein_fluct
    return s_ * np.log(1 + T_ * Bx**2 / s_) + (d_ - s_) * np.log(1 + Bx**2 / (d_ - s_) * bad)

L_val        = L_exact(T_PROB, sP, d, deltaP, delta0_prob)
C_radius_prob = By * np.sqrt(2 * L_val / alpha_prob) + Btheta
eps_prob      = (L_val**2 / T_PROB)**(1/3)
mu_prob       = eps_prob / (C_radius_prob + 1e-12)

def caseP_ctx(rng):
    x = np.zeros(d)
    in_A = rng.random() > deltaP
    if in_A:
        v = rng.standard_normal(sP); x[IP] = v / np.linalg.norm(v) * Bx
        return x, False
    else:
        v = rng.standard_normal(d - sP); x[IcP] = v / np.linalg.norm(v) * Bx
        return x, True

# ── Run all simulations ───────────────────────────────────────────────────────
T = T_SHORT
print("Running simulations (15 seeds × 8 cases) ...")
res1  = run_case('isotropic',    T, case1_ctx,  theta1,  mu1(T),  SEEDS)
res2  = run_case('single_dir',   T, case2_ctx,  theta2,  mu2(T),  SEEDS)
res3a = run_case('fixed_sparse', T, case3a_ctx, theta3a, mu3a(T), SEEDS)
res3b = run_case('var_sparse',   T, case3b_ctx, theta3b, mu3b(T), SEEDS)
res4  = run_case('global_cov',   T, case4_ctx,  theta4,  mu4(T),  SEEDS)
resA  = run_case('block_leakage',T, caseA_ctx,  thetaA,  muA(T),  SEEDS,
                 extra_tracking='block_eig')
resAb = run_case('block_abis',   T, caseAb_ctx, thetaAb, muAb(T), SEEDS)
resP  = run_case('prob_sub',  T_PROB, caseP_ctx, thetaP, mu_prob, SEEDS,
                 extra_tracking='prob_Ac')
print("  Done.")

RESULTS = {
    'isotropic':    (res1,  T_SHORT),
    'single_dir':   (res2,  T_SHORT),
    'fixed_sparse': (res3a, T_SHORT),
    'var_sparse':   (res3b, T_SHORT),
    'global_cov':   (res4,  T_SHORT),
    'block_leakage':(resA,  T_SHORT),
    'block_abis':   (resAb, T_SHORT),
    'prob_sub':     (resP,  T_PROB),
}

# =============================================================================
# GLOBAL FIGURES (4.1 – 4.5)
# =============================================================================

# ── Fig 4.1: logdet_all ───────────────────────────────────────────────────────
print("Plotting logdet_all.png ...")
fig, ax = plt.subplots(figsize=(12, 6))
t_ax = np.arange(1, T_SHORT + 1)
ax.plot(t_ax, d  * np.log(t_ax + 1), color='gray', linestyle='--',
        linewidth=1.0, label=r'$d\log t$')
ax.plot(t_ax, s3 * np.log(t_ax + 1), color='gray', linestyle='--',
        linewidth=1.0, label=r'$s\log t$')
for key in SHORT_KEYS:
    res, T_ = RESULTS[key]
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

# ── Fig 4.2: eigenvalues_grid ─────────────────────────────────────────────────
print("Plotting eigenvalues_grid.png ...")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
for idx, key in enumerate(SHORT_KEYS):
    ax = axes[idx]
    res, T_ = RESULTS[key]
    eigs_mean = res['eigs'].mean(0)
    t_ax = np.arange(1, T_ + 1)
    for j in range(d):
        col = COLORS[key] if eigs_mean[-1, j] > lam + 0.05 else '#aaaaaa'
        ax.semilogy(t_ax, eigs_mean[:, j], color=col, linewidth=0.9, alpha=0.75)
    if key == 'global_cov':
        cumG_mean = res['cumG'].mean(0)
        lmin_floor = lam + ell4 * cumG_mean / 2
        ax.semilogy(t_ax, lmin_floor, 'k--', linewidth=1.2,
                    label=r'$\lambda+\ell|G_t|/2$')
        ax.legend(fontsize=9)
    final_ld = res['logdet'].mean(0)[-1]
    ax.set_title(f"{NAMES[key]}\nfinal log det={final_ld:.1f}", fontsize=10)
    ax.set_xlabel('Round $t$', fontsize=9)
    ax.set_ylabel('Eigenvalue', fontsize=9)
    ax.tick_params(labelsize=8)
ax = axes[7]
res, T_p = RESULTS['prob_sub']
eigs_mean = res['eigs'].mean(0)
t_ax_p = np.arange(1, T_p + 1)
for j in range(d):
    col = COLORS['prob_sub'] if eigs_mean[-1, j] > lam + 0.05 else '#aaaaaa'
    ax.semilogy(t_ax_p, eigs_mean[:, j], color=col, linewidth=0.9, alpha=0.75)
final_ld = res['logdet'].mean(0)[-1]
ax.set_title(f"{NAMES['prob_sub']}\nfinal log det={final_ld:.1f}", fontsize=10)
ax.set_xlabel('Round $t$', fontsize=9)
ax.set_ylabel('Eigenvalue', fontsize=9)
ax.tick_params(labelsize=8)
fig.suptitle('Eigenvalue trajectories — all cases', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'eigenvalues_grid.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: eigenvalues_grid.png")

# ── lambda_min_global.png ─────────────────────────────────────────────────────
print("Plotting lambda_min_global.png ...")
res_gc, _ = RESULTS['global_cov']
lmin_mean = np.linalg.eigvalsh(
    np.eye(d)  # placeholder: recompute from eigs
)
# recompute lambda_min from stored eigenvalue trajectories
eigs_gc = res_gc['eigs'].mean(0)          # (T, d)
lmin_traj = eigs_gc.min(axis=1)           # (T,)
cumG_gc   = res_gc['cumG'].mean(0)        # (T,)
t_ax_gc   = np.arange(1, T + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(t_ax_gc, lmin_traj, color=COLORS['global_cov'], lw=2.2,
            label=r'$\lambda_{\min}(V_t)$ (empirical mean)')
floor = lam + ell4 * cumG_gc / 2
ax.semilogy(t_ax_gc, floor, 'k--', lw=1.8,
            label=r'$\lambda + \ell\,|G_t|/2$ (theoretical floor)')
ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel(r'$\lambda_{\min}(V_t)$', fontsize=12)
ax.set_title('Global covariance lower bound — minimum eigenvalue', fontsize=13)
ax.legend(fontsize=11)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'lambda_min_global.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: lambda_min_global.png")

# ── Fig 4.3: exploration_all ──────────────────────────────────────────────────
print("Plotting exploration_all.png ...")
res_p = resP
fig, ax = plt.subplots(figsize=(12, 6))
t_ax = np.arange(1, T_SHORT + 1)
for key in SHORT_KEYS:
    res, T_ = RESULTS[key]
    m = res['cumG'].mean(0); s_ = res['cumG'].std(0)
    ax.plot(t_ax, m, color=COLORS[key], label=NAMES[key], linewidth=1.4)
    ax.fill_between(t_ax, m - s_, m + s_, alpha=0.15, color=COLORS[key])
ax_in = ax.inset_axes([0.04, 0.52, 0.36, 0.44])
t_p = np.arange(1, T_PROB + 1)
m_p = res_p['cumG'].mean(0); s_p = res_p['cumG'].std(0)
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

# ── Fig 4.4: regret_all ───────────────────────────────────────────────────────
print("Plotting regret_all.png ...")
fig, ax = plt.subplots(figsize=(12, 6))
t_ax = np.arange(1, T_SHORT + 1)
for key in SHORT_KEYS:
    res, T_ = RESULTS[key]
    m = res['cumR'].mean(0); s_ = res['cumR'].std(0)
    ax.plot(t_ax, m, color=COLORS[key], label=NAMES[key], linewidth=1.4)
    ax.fill_between(t_ax, m - s_, m + s_, alpha=0.15, color=COLORS[key])
ax_in = ax.inset_axes([0.04, 0.52, 0.36, 0.44])
t_p = np.arange(1, T_PROB + 1)
m_p = res_p['cumR'].mean(0); s_p = res_p['cumR'].std(0)
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

# ── Fig 4.5: rate_check ───────────────────────────────────────────────────────
print("Plotting rate_check.png (runs extra short simulations) ...")
Ts_rate = [200, 400, 800, 1600, 3200]
seeds5  = list(range(5))
R_iso = []; R_fs = []
for T_ in Ts_rate:
    r = run_case('iso_r',  T_, case1_ctx,  theta1,  mu1(T_),  seeds5)
    R_iso.append(r['cumR'][:, -1].mean())
    r = run_case('fs_r',   T_, case3a_ctx, theta3a, mu3a(T_), seeds5)
    R_fs.append(r['cumR'][:, -1].mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
Ts_arr   = np.array(Ts_rate, dtype=float)
R_iso_a  = np.array(R_iso); R_fs_a = np.array(R_fs)
sl_iso   = np.polyfit(np.log(Ts_arr), np.log(np.maximum(R_iso_a, 1e-12)), 1)[0]
sl_fs    = np.polyfit(np.log(Ts_arr), np.log(np.maximum(R_fs_a,  1e-12)), 1)[0]

ax1.loglog(Ts_arr, R_iso_a, 'o-', color=COLORS['isotropic'],
           label=f'Isotropic (slope={sl_iso:.2f})', linewidth=1.4)
ax1.loglog(Ts_arr, R_fs_a,  's-', color=COLORS['fixed_sparse'],
           label=f'Fixed sparse (slope={sl_fs:.2f})', linewidth=1.4)
ref1 = R_iso_a[0] * (Ts_arr / Ts_arr[0])**(2/3)
ax1.loglog(Ts_arr, ref1, color='gray', linestyle='--', label='slope 2/3 ref')
ax1.set_xlabel('$T$', fontsize=12); ax1.set_ylabel('$R_T$', fontsize=12)
ax1.set_title('Regret vs $T$ (log-log)', fontsize=12)
ax1.legend(fontsize=10); ax1.tick_params(labelsize=10)

T_rate2 = 1600; s_vals = [1, 2, 3, 5, 7, 10]; R_s = []
for sv in s_vals:
    theta_sv = np.zeros(d); theta_sv[:sv] = 1 / np.sqrt(sv)
    def make_ctx(sv_=sv):
        def ctx(rng):
            x = np.zeros(d); v = rng.standard_normal(sv_)
            x[:sv_] = v / np.linalg.norm(v) * Bx; return x
        return ctx
    mu_sv = make_mu(T_rate2, (sv**2 * np.log(T_rate2)**2 / T_rate2)**(1/3),
                    C_rad(T_rate2, sv))
    r = run_case(f'fs_s{sv}', T_rate2, make_ctx(sv), theta_sv, mu_sv, seeds5)
    R_s.append(r['cumR'][:, -1].mean())

s_arr  = np.array(s_vals, dtype=float); R_s_a = np.array(R_s)
sl_s   = np.polyfit(np.log(s_arr), np.log(np.maximum(R_s_a, 1e-12)), 1)[0]
ax2.loglog(s_arr, R_s_a, 'D-', color=COLORS['fixed_sparse'],
           label=f'Fixed sparse (slope={sl_s:.2f})', linewidth=1.4)
ref2 = R_s_a[0] * (s_arr / s_arr[0])**(2/3)
ax2.loglog(s_arr, ref2, color='gray', linestyle='--', label='slope 2/3 ref')
ax2.set_xlabel('$s$', fontsize=12); ax2.set_ylabel('$R_T$', fontsize=12)
ax2.set_title(f'Regret vs $s$ at $T={T_rate2}$ (log-log)', fontsize=12)
ax2.legend(fontsize=10); ax2.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'rate_check.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: rate_check.png")

# =============================================================================
# ELLIPSOID FIGURES (4.6, 4.7, 4.8)
# Single-seed simulations saving V_t at four snapshot rounds.
# =============================================================================
SNAP_T = [50, 200, 500, 799]   # four time-points for all three cases

def run_ellipsoid_sim(sample_ctx_fn, theta_star_fn, mu_val, snap_rounds, seed=42,
                      prob_mode=False):
    """Run one seed, return dict of V_t snapshots and eigenvalue history."""
    rng   = np.random.default_rng(seed)
    V     = lam * np.eye(d)
    V_inv = (1/lam) * np.eye(d)
    sum_xy = np.zeros(d)
    snaps  = {}
    for t in range(T_SHORT):
        if prob_mode:
            x_t, _ = sample_ctx_fn(rng)
        else:
            x_t = sample_ctx_fn(rng)
        Vx   = V_inv @ x_t
        gate = np.sqrt(np.clip(x_t @ Vx, 0, None))
        if gate > mu_val:
            p_t   = rng.uniform(-By, By)
            y_t   = x_t @ theta_star_fn + rng.normal(0, noise_std)
            o_t   = 1.0 if p_t <= y_t else 0.0
            ytil  = 2 * By * (o_t - 0.5)
            denom = 1.0 + x_t @ Vx
            V_inv -= np.outer(Vx, Vx) / denom
            V     += np.outer(x_t, x_t)
            sum_xy += ytil * x_t
        if t in snap_rounds:
            snaps[t] = np.linalg.eigvalsh(V).copy()
    return snaps

def draw_ellipsoid_panels(fig, axes, snaps, snap_rounds, color_palette,
                          edge_color, active_label, frozen_label,
                          suptitle, active_eig_idx=-1, frozen_eig_idx=0):
    """Draw 4 ellipsoid panels. active_eig_idx=-1 uses largest eigenvalue."""
    colors = color_palette(np.linspace(0.40, 0.92, len(snap_rounds)))
    for k, t in enumerate(snap_rounds):
        ax   = axes[k]
        eigs = snaps.get(t, snaps[max(snaps.keys())])   # fallback to last
        lam_active = eigs[active_eig_idx]
        lam_frozen = eigs[frozen_eig_idx]
        lam_active = max(lam_active, 1e-6)
        lam_frozen = max(lam_frozen, 1e-6)
        # Confidence ellipsoid semi-axes: 1/sqrt(lambda)
        w = 2.0 / np.sqrt(lam_active)   # active: gets smaller over time
        h = 2.0 / np.sqrt(lam_frozen)   # frozen: stays near 2/sqrt(1) = 2
        ell = Ellipse([0, 0], w, h, angle=0,
                      facecolor=colors[k], alpha=0.60,
                      edgecolor=edge_color, linewidth=2.5)
        ax.add_patch(ell)
        lim = max(2.2, h * 0.7, w * 0.7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(0, color='#999999', lw=0.5)
        ax.axvline(0, color='#999999', lw=0.5)
        ax.set_title(f'$t = {t}$\n'
                     f'$\\lambda_{{\\mathrm{{act}}}}={lam_active:.1f}$\n'
                     f'$\\lambda_{{\\mathrm{{frz}}}}={lam_frozen:.2f}$',
                     fontsize=11)
        ax.set_xlabel(active_label, fontsize=11)
        if k == 0:
            ax.set_ylabel(frozen_label, fontsize=11)
        ax.tick_params(labelsize=9)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)

# ── Fig 4.6: Single direction ellipsoid ───────────────────────────────────────
print("Plotting ellipsoid_single_dir.png ...")
snaps2 = run_ellipsoid_sim(case2_ctx, theta2, mu2(T_SHORT), SNAP_T)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
draw_ellipsoid_panels(
    fig, axes, snaps2, SNAP_T,
    color_palette=plt.cm.Oranges,
    edge_color='darkorange',
    active_label=r'$e_1$ (explored)',
    frozen_label=r'$e_\perp$ (frozen at $\lambda$)',
    suptitle=(r'Single direction — Needle ellipsoid: $e_1$ shrinks, '
              r'all $d-1$ perpendicular axes frozen'),
    active_eig_idx=-1,
    frozen_eig_idx=0,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'ellipsoid_single_dir.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: ellipsoid_single_dir.png")

# ── Fig 4.7: Fixed sparse ellipsoid ───────────────────────────────────────────
print("Plotting ellipsoid_fixed_sparse.png ...")
snaps3a = run_ellipsoid_sim(case3a_ctx, theta3a, mu3a(T_SHORT), SNAP_T)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
draw_ellipsoid_panels(
    fig, axes, snaps3a, SNAP_T,
    color_palette=plt.cm.Purples,
    edge_color='#4dac26',
    active_label=r'Active ($S$)',
    frozen_label=r'Frozen ($S^c$)',
    suptitle=(rf'Fixed sparse support — Partial needle: $s={s3}$ active axes shrink, '
              rf'$d-s={d-s3}$ frozen'),
    active_eig_idx=-1,
    frozen_eig_idx=0,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'ellipsoid_fixed_sparse.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: ellipsoid_fixed_sparse.png")

# ── Fig 4.8: Variable sparse ellipsoid ────────────────────────────────────────
print("Plotting ellipsoid_var_sparse.png ...")
snaps3b = run_ellipsoid_sim(case3b_ctx, theta3b, mu3b(T_SHORT), SNAP_T)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
draw_ellipsoid_panels(
    fig, axes, snaps3b, SNAP_T,
    color_palette=plt.cm.Greens,
    edge_color='darkgreen',
    active_label=r'$e_1$',
    frozen_label=r'$e_2$',
    suptitle=(r'Variable sparse support — Near-spherical: no needle forms, '
              r'indistinguishable from isotropic baseline'),
    active_eig_idx=-1,
    frozen_eig_idx=0,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'ellipsoid_var_sparse.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: ellipsoid_var_sparse.png")

# =============================================================================
# DEDICATED CASE FIGURES (4.10 – 4.14)
# =============================================================================

# ── block_epl.png ─────────────────────────────────────────────────────────────
print("Plotting block_epl.png ...")
t_ax = np.arange(1, T_SHORT + 1)
extra_A    = resA['extra']
eig_I_all  = np.array([e['eig_I']  for e in extra_A])
eig_Ic_all = np.array([e['eig_Ic'] for e in extra_A])
fig, ax = plt.subplots(figsize=(11, 5))
m_I  = eig_I_all.mean(0);  s_I  = eig_I_all.std(0)
m_Ic = eig_Ic_all.mean(0); s_Ic = eig_Ic_all.std(0)
ax.plot(t_ax, m_I,  color=COLORS['block_leakage'],
        label='$I$-block gatekeeper contribution', linewidth=1.6)
ax.fill_between(t_ax, m_I - s_I,   m_I + s_I,   alpha=0.15, color=COLORS['block_leakage'])
ax.plot(t_ax, m_Ic, color='#e08214',
        label='$I^c$-block gatekeeper contribution', linewidth=1.6)
ax.fill_between(t_ax, m_Ic - s_Ic, m_Ic + s_Ic, alpha=0.15, color='#e08214')
ax.axhline(etaA**2, color='gray', linestyle='--', linewidth=1.2,
           label=r'$\eta^2 = 0.01$')
ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Gatekeeper contribution', fontsize=12)
ax.set_title('Block leakage — gatekeeper contributions', fontsize=13)
ax.legend(fontsize=10); ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'block_epl.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: block_epl.png")

# ── eta_necessary.png ─────────────────────────────────────────────────────────
print("Plotting eta_necessary.png ...")
resAb2 = run_case('block_abis_eig', T_SHORT, caseAb_ctx, thetaAb, muAb(T_SHORT),
                  SEEDS, extra_tracking='block_eig')
extra_Ab2  = resAb2['extra']
eig_Ic_Ab  = np.array([e['eig_Ic'] for e in extra_Ab2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
t_ax = np.arange(1, T_SHORT + 1)
m_Ab = eig_Ic_Ab.mean(0); s_Ab = eig_Ic_Ab.std(0)
ax1.plot(t_ax, m_Ab, color=COLORS['block_abis'], linewidth=1.6)
ax1.fill_between(t_ax, m_Ab - s_Ab, m_Ab + s_Ab, alpha=0.15, color=COLORS['block_abis'])
ax1.axhline(Bx**2 / 2, color='gray', linestyle='--', linewidth=1.2,
            label=r'$B_x^2/2 = 0.5$')
ax1.set_xlabel('Round $t$', fontsize=12)
ax1.set_ylabel(r'$I^c$-block gatekeeper contribution', fontsize=12)
ax1.set_title(r'Block-diagonal, no $\eta$ — $I^c$-block gatekeeper', fontsize=12)
ax1.legend(fontsize=10); ax1.tick_params(labelsize=10)

R_iso_f = res1['cumR'][:, -1].mean()
R_Ab_f  = resAb['cumR'][:, -1].mean()
bars = ax2.bar(['Isotropic', 'Block-diagonal'],
               [R_iso_f, R_Ab_f],
               color=[COLORS['isotropic'], COLORS['block_abis']], alpha=0.85)
ax2.set_ylabel('Final $R_T$', fontsize=12)
ax2.set_title(f'Regret comparison at $T={T_SHORT}$', fontsize=12)
ax2.tick_params(labelsize=10)
for bar, val in zip(bars, [R_iso_f, R_Ab_f]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.0, f'{val:.1f}',
             ha='center', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'eta_necessary.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: eta_necessary.png")

# ── block_diagonal_Vt.png ─────────────────────────────────────────────────────
print("Plotting block_diagonal_Vt.png ...")
rng_vt = np.random.default_rng(0)
V_vt   = lam * np.eye(d)
for _ in range(T_PROB):
    x_t, _ = caseP_ctx(rng_vt)
    Vx   = np.linalg.solve(V_vt, x_t)
    gate = np.sqrt(np.clip(x_t @ Vx, 0, None))
    if gate > mu_prob:
        V_vt += np.outer(x_t, x_t)
fig, ax = plt.subplots(figsize=(7, 6))
vmax = np.abs(V_vt).max()
im = ax.imshow(V_vt, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
ax.set_title('Block-diagonal structure of $V_T$\n'
             '(exact zeros off block, probability 1)', fontsize=12)
ax.set_xlabel('Coordinate index', fontsize=11)
ax.set_ylabel('Coordinate index', fontsize=11)
ax.tick_params(labelsize=10)
for pos in [sP - 0.5]:
    ax.axhline(pos, color='black', linewidth=1.8, linestyle='--')
    ax.axvline(pos, color='black', linewidth=1.8, linestyle='--')
ax.text(sP / 2 - 0.5, sP / 2 - 0.5, '$A$',
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(sP + (d - sP) / 2 - 0.5, sP + (d - sP) / 2 - 0.5, '$A^c$',
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'block_diagonal_Vt.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: block_diagonal_Vt.png")

# ── bernstein_concentration.png ──────────────────────────────────────────────
print("Plotting bernstein_concentration.png ...")
extra_P     = resP['extra']
cumG_Ac_all = np.array([e['cumG_Ac'] for e in extra_P])
cumG_all    = resP['cumG']
t_p = np.arange(1, T_PROB + 1)
m_GAc = cumG_Ac_all.mean(0)
m_G   = cumG_all.mean(0)
bernstein_bound = deltaP * m_G + np.sqrt(np.maximum(m_G, 0) * 4 * np.log(T_PROB) / 2)
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(t_p, m_GAc, color=COLORS['prob_sub'], linewidth=1.6,
        label=r'Empirical $|G_{A^c,t}|$ (mean, 15 seeds)')
ax.plot(t_p, bernstein_bound, color='#d73027', linestyle='--', linewidth=1.4,
        label=r'Bernstein bound $\delta|G_t|+\sqrt{2|G_t|\delta\log(1/\delta_0)}+\frac{2}{3}\log(1/\delta_0)$')
ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(r'Probabilistic subspace — Bernstein concentration of $|G_{A^c,t}|$',
             fontsize=13)
ax.legend(fontsize=10); ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'bernstein_concentration.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: bernstein_concentration.png")

# ── effective_L_regimes.png ───────────────────────────────────────────────────
print("Plotting effective_L_regimes.png ...")
delta_vals   = np.linspace(0.001, 1, 300)
delta0_fixed = T_PROB**(-4)
L_curve      = np.array([L_exact(T_PROB, sP, d, dv, delta0_fixed) for dv in delta_vals])
norm_regret  = L_curve**(2/3) * T_PROB**(2/3) / T_PROB
delta_star   = 2 * np.sqrt(np.log(T_PROB) / T_PROB)

fig, ax1 = plt.subplots(figsize=(11, 5.5))
ax1.plot(delta_vals, L_curve, color=COLORS['prob_sub'], linewidth=1.6,
         label=r'$\mathcal{L}(\delta)$')
ax1.axvline(deltaP,     color='#d73027', linestyle='--', linewidth=1.4,
            label=rf'$\delta={deltaP}$ (current)')
ax1.axvline(delta_star, color='#f4a582', linestyle=':',  linewidth=1.4,
            label=rf'$\delta^*=2\sqrt{{\log T/T}}\approx{delta_star:.3f}$')
ax1.set_xlabel(r'$\delta$', fontsize=12)
ax1.set_ylabel(r'$\mathcal{L}(\delta)$', fontsize=12, color=COLORS['prob_sub'])
ax1.tick_params(axis='y', labelcolor=COLORS['prob_sub'], labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax2 = ax1.twinx()
ax2.plot(delta_vals, norm_regret, color='#b2182b', linewidth=1.4, linestyle='-.',
         label=r'$\mathcal{L}^{2/3}/T^{1/3}$ (normalized regret)')
ax2.set_ylabel(r'$\mathcal{L}^{2/3}/T^{1/3}$', fontsize=12, color='#b2182b')
ax2.tick_params(axis='y', labelcolor='#b2182b', labelsize=10)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc='upper left')
ax1.set_title(rf'Effective $\mathcal{{L}}$ vs $\delta$ ($T={T_PROB}$, $s={sP}$, $d={d}$)',
              fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'effective_L_regimes.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved: effective_L_regimes.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"All figures saved to: {OUT}")
print("=" * 60)
figs = [
    'logdet_all.png', 'eigenvalues_grid.png', 'exploration_all.png',
    'regret_all.png', 'rate_check.png',
    'ellipsoid_single_dir.png', 'ellipsoid_fixed_sparse.png',
    'ellipsoid_var_sparse.png',
    'block_epl.png', 'eta_necessary.png', 'block_diagonal_Vt.png',
    'bernstein_concentration.png', 'effective_L_regimes.png',
]
for f in figs:
    status = "OK" if os.path.exists(os.path.join(OUT, f)) else "MISSING"
    print(f"  [{status}] {f}")
print("\nDone.")
