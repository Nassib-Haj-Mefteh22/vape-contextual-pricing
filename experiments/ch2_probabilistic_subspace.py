"""
============================================================
VAPE -- Probabilistic subspace: orthogonal A/Ac partition
============================================================

Setting:
    Two orthogonal subspaces A ⊂ R^s and A^c ⊂ R^{d-s}.
    P(X_t ∈ A^c | F_{t-1}) ≤ δ  a.s.   ||X_t|| ≤ Bx  a.s.
    NO covariance assumption. NO iid.

Exact effective log-dimension (Theorem 1):
    L(T,s,d,δ) = s log(1 + T Bx²/s)
               + (d-s) log(1 + Bx²/(d-s) * (δT + 2√(T log T)))
    (with α = δ₀ = T^{-4}, no approximation made)

Regret:
    E[R_T] ≲ L(T,s,d,δ)^{2/3} · T^{2/3} + O(T^{-3})

Five regimes:
    δ = 0 exactly      : L = s log T  → R_T ~ s^{2/3} T^{2/3}  (= Case 5)
    δ ≫ √(log T/T)    : signal dom.  → R_T ~ [s log T+(d-s)log(δT)]^{2/3} T^{2/3}
    δ ≪ √(log T/T)    : fluct. dom.  → R_T ~ [(s+d)/2 log T]^{2/3} T^{2/3}
    δ fixed, T large   : δT dominates → same as signal-dominates regime
    δ → 1              : all bad      → R_T ~ d^{2/3} T^{2/3}  (= Isotropic baseline)

Transition: T* = 4 log(T*)/δ²,  δ* = 2√(log T/T)

Figures generated:
    1. block_diagonal_Vt.png     -- V_t exact block structure, log det A and A^c tracked
    2. bernstein_concentration.png -- |G_{Ac}| vs δ|G|+√(2|G|δlog(1/δ₀))+2/3·log(1/δ₀); competing terms
    3. effective_L_regimes.png   -- L(T,s,d,δ) across regimes; transition at δ*=2√(logT/T)
    4. regret_interpolation.png  -- R_T for δ ∈ {0,small,medium,large,1}; log-log scaling
    5. criterion2_gatekeeper.png -- gatekeeper ||x_t||_{V^{-1}} vs μ, coloured by subspace
    6. transition_Tstar.png      -- T* = 4 log(T*)/δ² showing the two-regime boundary

Usage:
    python caseProbAAc.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OUT, exist_ok=True)
SEED = 42
rng  = np.random.default_rng(SEED)

# ============================================================
# PARAMETERS
# ============================================================
d     = 10
s     = 3
T     = 8000
delta = 0.10       # P(X_t in A^c | F_{t-1}) <= delta
Bx    = 1.0
By    = 1.0
Btheta = 1.0
lam   = 1.0
# Exact choices from Theorem 1
alpha  = T**(-4)
delta0 = T**(-4)

I  = list(range(s))
Ic = list(range(s, d))

theta_star      = np.zeros(d)
theta_star[:s]  = rng.standard_normal(s)
theta_star[:s] /= np.linalg.norm(theta_star[:s])

# ============================================================
# EXACT EFFECTIVE LOG-DIMENSION (Theorem 1, eq. 3)
# ============================================================
def L_exact(T, s, d, delta, delta0, Bx):
    """Exact formula, no approximation."""
    azuma_fluct = np.sqrt(T * np.log(1.0 / delta0) / 2.0)
    bad_count   = delta * T + azuma_fluct
    term_A  = s * np.log(1.0 + T * Bx**2 / s)
    term_Ac = (d - s) * np.log(1.0 + Bx**2 / (d - s) * bad_count)
    return term_A + term_Ac

L_val    = L_exact(T, s, d, delta, delta0, Bx)
C_radius = By * np.sqrt(2 * L_val / alpha) + Btheta
epsilon  = (L_val**2 / T) ** (1.0 / 3.0)
mu       = epsilon / (C_radius + 1e-12)

print(f"d={d}, s={s}, T={T}, delta={delta}")
print(f"L(T,s,d,delta) = {L_val:.3f}")
print(f"delta* = 2*sqrt(log T/T) = {2*np.sqrt(np.log(T)/T):.5f}  (transition)")
print(f"epsilon = {epsilon:.4f},  mu = {mu:.5f}")

# ============================================================
# CONTEXT SAMPLER
# ============================================================
def sample_context(rng, in_A):
    x = np.zeros(d)
    if in_A:
        v = rng.standard_normal(s)
        x[I] = v / np.linalg.norm(v) * Bx
    else:
        v = rng.standard_normal(d - s)
        x[Ic] = v / np.linalg.norm(v) * Bx
    return x

# ============================================================
# VAPE SIMULATION
# ============================================================
V_inv    = np.eye(d) / lam
sum_xy   = np.zeros(d)
theta_hat = np.zeros(d)

V_A_inv  = np.eye(s) / lam
V_Ac_inv = np.eye(d - s) / lam

logdet_A_hist  = []
logdet_Ac_hist = []
gate_hist      = []
branch1_rounds = np.zeros(T, dtype=int)
in_A_hist      = np.zeros(T, dtype=int)
bad_b1_cumul   = np.zeros(T, dtype=int)
G_cum          = np.zeros(T, dtype=int)
cum_regret     = np.zeros(T)

G_total = 0; G_Ac_total = 0; regret_total = 0.0

for t in range(T):
    in_A = (rng.random() >= delta)
    x_t  = sample_context(rng, in_A)
    in_A_hist[t] = int(in_A)

    Vx       = V_inv @ x_t
    gate_val = float(np.sqrt(x_t @ Vx))
    gate_hist.append(gate_val)

    if gate_val >= mu:
        G_total += 1
        branch1_rounds[t] = 1
        if not in_A:
            G_Ac_total += 1

        p_t = rng.uniform(-By, By)
        y_t = x_t @ theta_star + rng.standard_normal() * 0.05
        o_t = int(y_t >= p_t)
        iota = 2 * By
        sum_xy += iota * (o_t - 0.5) * x_t
        V_inv  -= np.outer(Vx, Vx) / (1.0 + x_t @ Vx)

        if in_A:
            xI = x_t[I]
            Vx_A = V_A_inv @ xI
            V_A_inv -= np.outer(Vx_A, Vx_A) / (1.0 + xI @ Vx_A)
        else:
            xIc = x_t[Ic]
            Vx_Ac = V_Ac_inv @ xIc
            V_Ac_inv -= np.outer(Vx_Ac, Vx_Ac) / (1.0 + xIc @ Vx_Ac)

        theta_hat = V_inv @ sum_xy
        regret_total += epsilon
    else:
        p_t = np.clip(x_t @ theta_hat, -By, By)
        y_t = x_t @ theta_star + rng.standard_normal() * 0.05
        regret_total += max(0.0, abs(y_t - p_t))

    cum_regret[t] = regret_total
    G_cum[t] = G_total
    bad_b1_cumul[t] = G_Ac_total

    _, ld_A  = np.linalg.slogdet(np.linalg.inv(V_A_inv))
    _, ld_Ac = np.linalg.slogdet(np.linalg.inv(V_Ac_inv))
    logdet_A_hist.append(ld_A)
    logdet_Ac_hist.append(ld_Ac)

gate_hist      = np.array(gate_hist)
logdet_A_hist  = np.array(logdet_A_hist)
logdet_Ac_hist = np.array(logdet_Ac_hist)
t_arr = np.arange(1, T + 1)

print(f"\nBranch-1 rounds: {G_total}/{T}")
print(f"|G_{{Ac}}| = {G_Ac_total}  (expected <= {delta*G_total:.1f} + {np.sqrt(G_total*np.log(1/delta0)/2):.1f})")
print(f"Azuma bound: {delta*G_total + np.sqrt(G_total*np.log(1/delta0)/2):.1f}")
print(f"Final cumulative regret: {cum_regret[-1]:.2f}")

# ============================================================
# FIG 1 -- Block-diagonal V_t
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t_arr, logdet_A_hist,  color='steelblue', label=r'$\log\det V^A_t$ (A-block)')
axes[0].plot(t_arr, logdet_Ac_hist, color='tomato', linestyle='--', label=r'$\log\det V^{A^c}_t$ (A$^c$-block)')
axes[0].plot(t_arr, logdet_A_hist + logdet_Ac_hist, color='purple', linewidth=1.8, label=r'$\log\det V_t$ (total)')

# Exact L bound
L_arr = np.array([L_exact(t, s, d, delta, delta0, Bx) for t in range(1, T+1)])
axes[0].plot(t_arr, L_arr, 'k:', linewidth=1.5, label=r'$\mathcal{L}(t)$ bound')
axes[0].set_xlabel('Round $t$')
axes[0].set_ylabel(r'$\log\det$')
axes[0].set_title('Block-diagonal decomposition of $V_t$\n(exact, pathwise, probability 1)')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

V_final = np.linalg.inv(V_inv)
im = axes[1].imshow(np.abs(V_final), cmap='Blues', aspect='auto')
axes[1].axhline(s - 0.5, color='red', lw=1.5, ls='--')
axes[1].axvline(s - 0.5, color='red', lw=1.5, ls='--')
axes[1].text(s/2,     s/2,          'A-block',      ha='center', va='center', color='red', fontsize=9)
axes[1].text(s+(d-s)/2, s+(d-s)/2, r'$A^c$-block', ha='center', va='center', color='red', fontsize=9)
plt.colorbar(im, ax=axes[1])
axes[1].set_title(r'$|V_T|$ heatmap — exact zeros off-block')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'block_diagonal_Vt.png'), dpi=120); plt.close()
print("Saved block_diagonal_Vt.png")

# ============================================================
# FIG 2 -- Azuma concentration + the two competing terms
# ============================================================
azuma_bound = delta * G_cum + np.sqrt(np.maximum(G_cum, 1) * np.log(1.0 / delta0) / 2.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t_arr, bad_b1_cumul, color='tomato', label=r'$|G_{A^c,t}|$ observed')
axes[0].plot(t_arr, delta * G_cum, color='orange', ls='--', label=r'$\delta|G_t|$ (expected)')
axes[0].plot(t_arr, azuma_bound, color='darkred', ls=':', label=r'$\delta|G_t|+\sqrt{|G_t|\log(1/\delta_0)/2}$ (Azuma)')
axes[0].set_xlabel('Round $t$'); axes[0].set_ylabel('Count')
axes[0].set_title('Azuma-Hoeffding concentration')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# The two competing terms in L: δT vs 2√(T log T)
T_range = np.logspace(2, 5, 300)
term_signal  = delta * T_range
term_fluct   = 2 * np.sqrt(T_range * np.log(T_range))
# T* where they are equal
def f_Tstar(T_): return delta * T_ - 2 * np.sqrt(T_ * np.log(T_))
try:
    T_star = brentq(f_Tstar, 10, 1e7)
except Exception:
    T_star = None

axes[1].loglog(T_range, term_signal, color='steelblue', linewidth=2, label=r'$\delta T$ (signal term)')
axes[1].loglog(T_range, term_fluct,  color='tomato',    linewidth=2, label=r'$2\sqrt{T\log T}$ (Azuma fluct.)')
if T_star:
    axes[1].axvline(T_star, color='black', ls=':', label=rf'$T^* \approx {T_star:.0f}$')
    axes[1].text(T_star*1.1, term_signal[0]*2, rf'$T^*$', fontsize=9)
axes[1].axvline(T, color='gray', ls='--', alpha=0.6, label=f'Current T={T}')
axes[1].set_xlabel('Horizon $T$'); axes[1].set_ylabel('Magnitude')
axes[1].set_title(r'Two competing terms in $\mathcal{L}$' + '\n' + rf'($\delta={delta}$, transition $T^* = 4\log T^*/\delta^2$)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, which='both')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'bernstein_concentration.png'), dpi=120); plt.close()
print("Saved bernstein_concentration.png")

# ============================================================
# FIG 3 -- L(T,s,d,delta) across all regimes vs delta
# ============================================================
delta_vals = np.logspace(-5, 0, 400)
delta_star = 2 * np.sqrt(np.log(T) / T)

L_vals = np.array([L_exact(T, s, d, dv, delta0, Bx) for dv in delta_vals])
R_vals = L_vals**(2/3) * T**(2/3)

# reference lines
L_case5 = s * np.log(1 + T * Bx**2 / s)                    # δ=0 (Case 5)
L_case1 = d * np.log(1 + T * Bx**2 / d)                    # δ→1 (Case 1)
L_fluct  = s * np.log(1 + T * Bx**2 / s) + (d-s) * np.log(1 + Bx**2/(d-s) * 2*np.sqrt(T*np.log(T)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogx(delta_vals, L_vals, color='purple', lw=2, label=r'$\mathcal{L}(T,s,d,\delta)$ exact')
axes[0].axhline(L_case5, color='steelblue', ls='--', label=r'$\delta=0$: $s\log T$ (Block-diagonal joint sparsity)')
axes[0].axhline(L_case1, color='tomato',    ls='--', label=r'$\delta=1$: $d\log T$ (Isotropic baseline)')
axes[0].axhline(L_fluct, color='green',     ls=':',  label=r'fluctuation-dominated: $\frac{s+d}{2}\log T$')
axes[0].axvline(delta_star, color='black', ls=':', lw=1.2, label=rf'$\delta^*=2\sqrt{{\log T/T}}\approx{delta_star:.4f}$')
axes[0].set_xlabel(r'$\delta$'); axes[0].set_ylabel(r'$\mathcal{L}$')
axes[0].set_title(r'Effective log-dimension $\mathcal{L}(T,s,d,\delta)$' + '\n' + 'five distinct regimes')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].semilogx(delta_vals, R_vals, color='purple', lw=2, label=r'$\mathcal{L}^{2/3}T^{2/3}$')
axes[1].axhline(L_case5**(2/3)*T**(2/3), color='steelblue', ls='--', label=r'Block-diagonal (joint sparsity): $s^{2/3}T^{2/3}$')
axes[1].axhline(L_case1**(2/3)*T**(2/3), color='tomato',    ls='--', label=r'Isotropic baseline: $d^{2/3}T^{2/3}$')
axes[1].axvline(delta_star, color='black', ls=':', lw=1.2)
axes[1].set_xlabel(r'$\delta$'); axes[1].set_ylabel('Regret bound')
axes[1].set_title('Regret bound interpolation')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'effective_L_regimes.png'), dpi=120); plt.close()
print("Saved effective_L_regimes.png")

# ============================================================
# FIG 4 -- Cumulative regret for multiple delta values
# ============================================================
delta_list = [0.0, delta_star, 0.05, 0.15, 0.5]
labels_d   = [r'$\delta=0$ (Block-diagonal joint sparsity)', rf'$\delta=\delta^*={delta_star:.4f}$', r'$\delta=0.05$',
              r'$\delta=0.15$', r'$\delta=0.5$']
colors_d   = ['steelblue', 'green', 'gold', 'orange', 'tomato']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for dv, col, lab in zip(delta_list, colors_d, labels_d):
    rng2 = np.random.default_rng(SEED + 1)
    d0   = max(T**(-4), 1e-15)
    Lv   = L_exact(T, s, d, dv, d0, Bx) if dv > 0 else s * np.log(1 + T * Bx**2 / s)
    Cr   = By * np.sqrt(2 * Lv / max(T**(-4), 1e-15)) + Btheta
    eps  = (Lv**2 / T) ** (1/3)
    muv  = eps / (Cr + 1e-12)

    V2 = np.eye(d) / lam; sxy2 = np.zeros(d); th2 = np.zeros(d); reg2 = 0.0; cum2 = []
    for t in range(T):
        inA2 = (rng2.random() >= dv)
        xt2  = sample_context(rng2, inA2)
        Vx2  = V2 @ xt2
        gv2  = float(np.sqrt(xt2 @ Vx2))
        if gv2 >= muv:
            p2 = rng2.uniform(-By, By)
            y2 = xt2 @ theta_star + rng2.standard_normal() * 0.05
            o2 = int(y2 >= p2)
            sxy2 += 2*By*(o2-0.5)*xt2
            V2   -= np.outer(Vx2, Vx2) / (1.0 + xt2 @ Vx2)
            th2   = V2 @ sxy2; reg2 += eps
        else:
            p2 = np.clip(xt2 @ th2, -By, By)
            y2 = xt2 @ theta_star + rng2.standard_normal() * 0.05
            reg2 += max(0.0, abs(y2 - p2))
        cum2.append(reg2)
    axes[0].plot(np.arange(T), cum2, color=col, label=lab, lw=1.2)

axes[0].set_xlabel('Round $t$'); axes[0].set_ylabel('Cumulative regret')
axes[0].set_title('Cumulative regret for varying $\\delta$')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

T_range2 = np.logspace(2, np.log10(T), 200)
for dv, col, lab in zip(delta_list, colors_d, labels_d):
    d0  = max(T**(-4), 1e-15)
    Lv2 = np.array([L_exact(t, s, d, dv, d0, Bx) if dv>0 else s*np.log(1+t*Bx**2/s) for t in T_range2])
    axes[1].loglog(T_range2, Lv2**(2/3)*T_range2**(2/3), color=col, label=lab, lw=1.2)

axes[1].set_xlabel('Horizon $T$'); axes[1].set_ylabel(r'$\mathcal{L}^{2/3}T^{2/3}$')
axes[1].set_title('Theoretical regret bound vs $T$ (log-log)')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3, which='both')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'regret_interpolation.png'), dpi=120); plt.close()
print("Saved regret_interpolation.png")

# ============================================================
# FIG 5 -- Gatekeeper criterion
# ============================================================
from matplotlib.lines import Line2D
subset = min(T, 3000)
c_arr  = np.where(in_A_hist[:subset] == 1, 'steelblue', 'tomato')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(range(subset), gate_hist[:subset], c=c_arr, s=2, alpha=0.4)
axes[0].axhline(mu, color='black', lw=1.5, ls='--', label=rf'$\mu={mu:.5f}$')
axes[0].set_xlabel('Round $t$'); axes[0].set_ylabel(r'$\|x_t\|_{V_t^{-1}}$')
axes[0].set_title('Gatekeeper: blue=$X_t\\in A$, red=$X_t\\in A^c$')
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor='steelblue',label='$X_t\\in A$'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='tomato',label='$X_t\\in A^c$'),
       Line2D([0],[0],color='black',ls='--',label=rf'$\mu={mu:.5f}$')]
axes[0].legend(handles=leg, fontsize=8); axes[0].grid(True, alpha=0.3)

gA  = gate_hist[:subset][in_A_hist[:subset]==1]
gAc = gate_hist[:subset][in_A_hist[:subset]==0]
bins = np.linspace(0, gate_hist[:subset].max()*1.05, 60)
axes[1].hist(gA,  bins=bins, alpha=0.5, color='steelblue', label='$X_t\\in A$', density=True)
axes[1].hist(gAc, bins=bins, alpha=0.5, color='tomato',    label='$X_t\\in A^c$', density=True)
axes[1].axvline(mu, color='black', lw=1.5, ls='--', label=rf'$\mu={mu:.5f}$')
axes[1].set_xlabel(r'$\|x_t\|_{V_t^{-1}}$'); axes[1].set_ylabel('Density')
axes[1].set_title('Gatekeeper distribution by subspace')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'criterion2_gatekeeper.png'), dpi=120); plt.close()
print("Saved criterion2_gatekeeper.png")

# ============================================================
# FIG 6 -- T* transition boundary
# ============================================================
delta_range = np.logspace(-3, -0.3, 300)
T_star_vals = []
for dv in delta_range:
    try:
        Ts = brentq(lambda T_: dv*T_ - 2*np.sqrt(T_*np.log(T_)), 2, 1e9)
        T_star_vals.append(Ts)
    except Exception:
        T_star_vals.append(np.nan)
T_star_vals = np.array(T_star_vals)

# Approximate: T* ≈ 4 log(T*)/δ², good approx: T* ≈ 4 log(4/δ²)/δ²
T_star_approx = 4 * np.log(4 / delta_range**2) / delta_range**2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].loglog(delta_range, T_star_vals,   color='purple', lw=2, label=r'$T^*$ exact')
axes[0].loglog(delta_range, T_star_approx, color='gray', ls='--', lw=1.5, label=r'$4\log(4/\delta^2)/\delta^2$')
axes[0].axhline(T, color='steelblue', ls=':', lw=1.5, label=f'Current $T={T}$')
axes[0].set_xlabel(r'$\delta$'); axes[0].set_ylabel(r'$T^*$')
axes[0].set_title(r'Transition $T^*(\delta)$: below = fluctuation-dominated' + '\n' + 'above = signal-dominated')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, which='both')

# δ* as function of T
T_range3 = np.logspace(2, 6, 200)
dstar_vals = 2 * np.sqrt(np.log(T_range3) / T_range3)
axes[1].loglog(T_range3, dstar_vals, color='purple', lw=2, label=r'$\delta^* = 2\sqrt{\log T/T}$')
axes[1].axvline(T, color='steelblue', ls=':', lw=1.5, label=f'$T={T}$')
axes[1].axhline(delta, color='tomato', ls='--', lw=1.5, label=rf'Current $\delta={delta}$')
# mark operating point
dstar_now = 2 * np.sqrt(np.log(T) / T)
regime = "signal" if delta > dstar_now else "fluctuation"
axes[1].scatter([T], [dstar_now], color='black', zorder=5, s=60)
axes[1].text(T*1.15, dstar_now*1.3, rf'$\delta^*={dstar_now:.4f}$' + f'\n({regime}-dom.)', fontsize=8)
axes[1].set_xlabel('Horizon $T$'); axes[1].set_ylabel(r'$\delta^*$')
axes[1].set_title(r'Regime boundary $\delta^*(T) = 2\sqrt{\log T / T}$')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, which='both')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'transition_Tstar.png'), dpi=120); plt.close()
print("Saved transition_Tstar.png")

print(f"\nAll 6 figures saved to {OUT}")
