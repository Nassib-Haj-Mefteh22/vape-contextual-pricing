"""
chapter3_adversarial_B.py
=========================
Setting B: adversarial contexts + non-Lipschitz (cliff) demand.

Theoretical roles
-----------------
CMRUP    Stage-1 estimator trained on e1 only -> theta_hat[1] = 0 ->
         g_hat(e2) = 0 while g_true(e2) = 1 -> constant per-round
         mispricing -> near-linear regret.

VAPE     PE phase uses successive elimination with Lipschitz correction
         (2 * L_xi * eps). For cliff demand (D is a step function,
         not Lipschitz) the correction is invalid: it is so large that
         the confidence intervals for all arms overlap permanently ->
         arm elimination never fires -> round-robin exploration ->
         near-linear regret.

VAPE-NL  PE phase does NOT assume Lipschitz demand. Direct greedy UCB
         on estimated demand, no Lipschitz correction -> the demand
         step between adjacent arms is detected quickly -> converges to
         the near-optimal arm -> sub-linear regret.

Produces: ch3_adversarial_B.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
from scipy.stats import linregress

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OUT, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
d       = 5
Bx      = 1.0
By      = 2.0
Btheta  = 1.0
Bxi     = 0.5
lam     = 1.0
MU      = 0.05      # gatekeeper threshold (same as Setting A)
C_PE    = 0.5       # UCB bonus coefficient for VAPE-NL PE
N_SEEDS = 15
SEEDS   = list(range(N_SEEDS))

theta_star    = np.zeros(d)
theta_star[1] = 1.0         # CMRUP Stage-1 (all e1) is blind to this

T_ADV = [2_000, 8_000, 32_000, 128_000]

COL = {'vape': '#d6604d', 'cmrup': '#2166ac', 'vape_nl': '#4dac26'}
LAB = {'vape': 'VAPE (Lip. PE)', 'cmrup': 'CMRUP', 'vape_nl': 'VAPE-NL (no Lip.)'}

# ── Cliff noise ────────────────────────────────────────────────────────────────
# xi in {-Bxi/2, +Bxi/2} with equal probability — step-function CDF, not Lipschitz
def sample_noise(rng):
    return (Bxi / 2.0) * (1.0 if rng.random() < 0.5 else -1.0)

def demand(w):
    """D(w) = P(xi >= w).  Cliff: jumps of 0.5 at w = ±Bxi/2."""
    if w <= -Bxi / 2.0:
        return 1.0
    elif w <= Bxi / 2.0:
        return 0.5
    else:
        return 0.0

def oracle_revenue(g):
    """Analytical max_{p>=0} p * D(p-g) for cliff noise."""
    r_below = max(0.0, g - Bxi / 2.0) * 1.0   # price = g - Bxi/2, demand = 1
    r_above = max(0.0, g + Bxi / 2.0) * 0.5   # price = g + Bxi/2, demand = 0.5
    return max(r_below, r_above)

# ── Adversarial context ────────────────────────────────────────────────────────
def ctx(t, T, rng):
    T1 = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    x  = np.zeros(d)
    x[0 if t < T1 else 1] = Bx
    return x

# ── Shared helpers ─────────────────────────────────────────────────────────────
def vape_eps(T):
    return (d ** 2 * np.log(T) ** 2 / T) ** (1.0 / 3.0)

def cmrup_Delta(T):
    T1 = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    return max(1e-6, 0.08 * By * Bx ** 2 * np.sqrt(d * np.log(T) / T1))

# ── VA phase (shared by VAPE and VAPE-NL) ─────────────────────────────────────
def va_step(x_t, g_t, V_inv, V, sum_xy, theta_hat, rng):
    """One VA round: random price, Sherman-Morrison update, return p_posted."""
    p_t  = rng.uniform(-By, By)
    xi_t = sample_noise(rng)
    o_t  = float(p_t <= g_t + xi_t)
    ytil = o_t - 0.5                    # unbiased for g_t / (2*By)
    Vx   = V_inv @ x_t
    V_inv -= np.outer(Vx, Vx) / (1.0 + float(x_t @ Vx))
    V    += np.outer(x_t, x_t)
    sum_xy    += ytil * x_t
    theta_hat[:] = 2.0 * By * (V_inv @ sum_xy)
    return p_t

# ── VAPE: successive elimination with Lipschitz correction ────────────────────
def run_vape(T, seed):
    """
    PE uses Lipschitz constant L_xi = 1/(2*Bxi), valid for uniform noise.
    Cliff demand is NOT Lipschitz: the correction 2*L_xi*eps is so large
    that the UCB/LCB intervals of all valid arms overlap permanently ->
    arm elimination never fires -> round-robin -> linear regret.
    """
    rng  = np.random.default_rng(seed)
    eps  = vape_eps(T)
    Lxi  = 1.0 / (2.0 * Bxi)    # assumes Lipschitz CDF -- violated by cliff
    alph = max(T ** (-4), 1e-300)

    K_max = int(np.ceil((By + Btheta) / eps)) + 2
    K_arr = np.arange(-K_max, K_max + 1)
    n_k   = len(K_arr)

    V     = lam * np.eye(d)
    V_inv = np.eye(d) / lam
    sum_xy    = np.zeros(d)
    theta_hat = np.zeros(d)
    N_pe = np.zeros(n_k, dtype=np.int64)
    D_pe = np.zeros(n_k)

    R = 0.0
    for t in range(T):
        x_t   = ctx(t, T, rng)
        g_t   = float(x_t @ theta_star)
        g_hat = float(x_t @ theta_hat)
        gate  = np.sqrt(max(float(x_t @ (V_inv @ x_t)), 0.0))

        if gate > MU:
            p_posted = va_step(x_t, g_t, V_inv, V, sum_xy, theta_hat, rng)
        else:
            prices = g_hat + K_arr * eps
            A = np.where((prices >= 0.0) & (prices <= By))[0]
            if len(A) == 0:
                A = np.array([np.argmin(np.abs(prices - np.clip(g_hat, 0.0, By)))])

            N_A = N_pe[A];  D_A = D_pe[A];  p_A = prices[A]
            cb  = np.sqrt(2.0 * np.log(1.0 / alph) / np.maximum(N_A, 1))

            # Lipschitz correction: INVALID for cliff demand
            UCB = p_A * np.minimum(1.0, D_A + cb + 2.0 * Lxi * eps)
            LCB = p_A * np.maximum(0.0, D_A - cb - 2.0 * Lxi * eps)
            best_lcb  = float(np.max(LCB))
            surviving = A[UCB >= best_lcb]
            # round-robin among surviving arms (all survive for cliff noise)
            k_sel    = surviving[np.argmin(N_pe[surviving])]
            p_posted = float(prices[k_sel])

            xi_t = sample_noise(rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_pe[k_sel] += 1
            D_pe[k_sel] += (o_t - D_pe[k_sel]) / N_pe[k_sel]

        R += oracle_revenue(g_t) - p_posted * demand(p_posted - g_t)
    return R

# ── VAPE-NL: greedy UCB, no Lipschitz assumption ──────────────────────────────
def run_vape_nl(T, seed):
    """
    PE uses plain greedy UCB with no Lipschitz correction.
    Correctly detects the demand step between adjacent arms and converges
    to the near-optimal arm (k = -1, revenue = 0.7 vs oracle 0.75).
    """
    rng  = np.random.default_rng(seed)
    eps  = vape_eps(T)

    K_max = int(np.ceil((By + Btheta) / eps)) + 2
    K_arr = np.arange(-K_max, K_max + 1)
    n_k   = len(K_arr)

    V     = lam * np.eye(d)
    V_inv = np.eye(d) / lam
    sum_xy    = np.zeros(d)
    theta_hat = np.zeros(d)
    N_pe = np.zeros(n_k, dtype=np.int64)
    D_pe = np.zeros(n_k)

    R = 0.0
    for t in range(T):
        x_t   = ctx(t, T, rng)
        g_t   = float(x_t @ theta_star)
        g_hat = float(x_t @ theta_hat)
        gate  = np.sqrt(max(float(x_t @ (V_inv @ x_t)), 0.0))

        if gate > MU:
            p_posted = va_step(x_t, g_t, V_inv, V, sum_xy, theta_hat, rng)
        else:
            prices = g_hat + K_arr * eps
            A = np.where((prices >= 0.0) & (prices <= By))[0]
            if len(A) == 0:
                A = np.array([np.argmin(np.abs(prices - np.clip(g_hat, 0.0, By)))])

            p_A = prices[A]
            # No Lipschitz correction: direct UCB on estimated demand
            ucb = p_A * np.minimum(1.0,
                    D_pe[A] + C_PE * np.sqrt(np.log(T) / np.maximum(N_pe[A], 1)))
            k_sel    = A[np.argmax(ucb)]
            p_posted = float(prices[k_sel])

            xi_t = sample_noise(rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_pe[k_sel] += 1
            D_pe[k_sel] += (o_t - D_pe[k_sel]) / N_pe[k_sel]

        R += oracle_revenue(g_t) - p_posted * demand(p_posted - g_t)
    return R

# ── CMRUP ──────────────────────────────────────────────────────────────────────
def run_cmrup(T, seed):
    """
    Standard CMRUP. Stage-1 contexts are all e1 -> theta_hat[1] = 0 ->
    g_hat(e2) = 0 (true g = 1) -> systematic mispricing in Stage 3.
    """
    rng   = np.random.default_rng(seed)
    T1    = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    Tw    = int(np.ceil(T ** (2.0 / 3.0)))
    Delta = cmrup_Delta(T)
    M_c   = max(1, int(np.ceil(Bxi / Delta)))
    w_grid = np.array([-Bxi + 2.0 * j * Delta for j in range(M_c + 1)])

    N_c = np.zeros(M_c, dtype=np.int64)
    D_c = np.zeros(M_c)
    theta_hat = np.zeros(d)
    X_buf = np.zeros((T1, d))
    z_buf = np.zeros(T1)

    R = 0.0
    for t in range(T):
        x_t = ctx(t, T, rng)
        g_t = float(x_t @ theta_star)

        if t < T1:
            p_t  = rng.uniform(-By, By)
            xi_t = sample_noise(rng)
            o_t  = float(p_t <= g_t + xi_t)
            X_buf[t] = x_t
            z_buf[t] = 2.0 * By * (o_t - 0.5)
            if t == T1 - 1:
                XtX = X_buf.T @ X_buf + lam * np.eye(d)
                theta_hat = np.linalg.solve(XtX, X_buf.T @ z_buf)
            p_posted = p_t

        elif t < T1 + Tw:
            g_hat = float(x_t @ theta_hat)
            j = (t - T1) % M_c
            p_posted = float(np.clip(g_hat + w_grid[j + 1] - 3.0 * Delta, 0.0, By))
            xi_t = sample_noise(rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_c[j] += 1
            D_c[j] += (o_t - D_c[j]) / N_c[j]

        else:
            g_hat = float(x_t @ theta_hat)
            b     = 0.5 * np.sqrt(np.log(T) / np.maximum(N_c, 1))
            UCB   = (g_hat + w_grid[1:] + Delta) * np.minimum(1.0, D_c + b)
            j_t   = int(np.argmax(UCB))
            a_t   = (j_t - 1 if b[j_t] <= Delta and j_t >= 1
                     else (0 if b[j_t] <= Delta else j_t))
            p_posted = float(np.clip(g_hat + w_grid[a_t + 1] - 3.0 * Delta, 0.0, By))
            xi_t = sample_noise(rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_c[a_t] += 1
            D_c[a_t] += (o_t - D_c[a_t]) / N_c[a_t]

        R += oracle_revenue(g_t) - p_posted * demand(p_posted - g_t)
    return R

# ── Experiment ─────────────────────────────────────────────────────────────────
ALGOS = [('vape', run_vape), ('cmrup', run_cmrup), ('vape_nl', run_vape_nl)]

def run_all(T_values):
    res   = {a: {T: np.zeros(N_SEEDS) for T in T_values} for a, _ in ALGOS}
    total = len(T_values) * len(ALGOS) * N_SEEDS
    done  = 0
    t0    = time.time()
    for T in T_values:
        for name, fn in ALGOS:
            for si, seed in enumerate(SEEDS):
                res[name][T][si] = fn(T, seed)
                done += 1
                if done % 10 == 0:
                    el  = time.time() - t0
                    eta = el / done * (total - done)
                    print(f"  {done}/{total}  ({el:.0f}s, ~{eta:.0f}s left)")
    return res

def slope(T_values, means):
    s, *_ = linregress(np.log(np.array(T_values, float)),
                       np.log(np.maximum(np.array(means, float), 1e-12)))
    return s

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 62)
    print("Setting B: adversarial contexts + cliff (non-Lipschitz) noise")
    print("=" * 62)
    res = run_all(T_ADV)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    order = ['vape_nl', 'cmrup', 'vape']   # VAPE-NL first (legend top)
    for algo in order:
        Rmat = np.array([res[algo][T] for T in T_ADV]).T
        mu_r = Rmat.mean(0)
        sig  = Rmat.std(0)
        sl   = slope(T_ADV, mu_r)
        ax.plot(T_ADV, mu_r, 'o-', color=COL[algo], lw=2.2,
                label=f"{LAB[algo]}  (slope = {sl:.2f})")
        ax.fill_between(T_ADV, np.maximum(mu_r - sig, 1e-3), mu_r + sig,
                        color=COL[algo], alpha=0.20)

    # T^{2/3} reference anchored to VAPE-NL
    T_arr   = np.array(T_ADV, float)
    ref_lvl = res['vape_nl'][T_ADV[-1]].mean()
    ax.plot(T_ADV, ref_lvl * (T_arr / T_ADV[-1]) ** (2.0 / 3.0),
            'k--', lw=0.9, alpha=0.5, label=r'$T^{2/3}$ reference')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Horizon $T$', fontsize=12)
    ax.set_ylabel('Cumulative regret $R_T$', fontsize=12)
    ax.set_title('Setting B: adversarial contexts, non-Lipschitz demand', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    note = (
        r'$\theta^*=e_2$; adversarial context $x_t=e_1\to e_2$ at $t=\lceil T^{2/3}\rceil$.' '\n'
        r'Cliff noise $\xi_t\in\{-B_\xi/2,\,+B_\xi/2\}$, $B_\xi=0.5$ (non-Lipschitz CDF).' '\n'
        r'VAPE Lip. PE: successive elimination with $L_\xi$ correction — invalid for cliff.' '\n'
        r'VAPE-NL: greedy UCB, no Lipschitz assumption — correctly identifies demand step.'
    )
    ax.annotate(note, xy=(0.03, 0.97), xycoords='axes fraction',
                fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85))

    fig.tight_layout()
    outpath = os.path.join(OUT, 'ch3_adversarial_B.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: ch3_adversarial_B.png")

    print("\n" + "=" * 62)
    print(f"SUMMARY  (T={T_ADV[-1]:,}, mean ± std, {N_SEEDS} seeds)")
    print("=" * 62)
    for algo, _ in ALGOS:
        arr = res[algo][T_ADV[-1]]
        sl  = slope(T_ADV, np.array([res[algo][T] for T in T_ADV]).mean(1))
        print(f"  {LAB[algo]:28s}: R_T = {arr.mean():.1f} ± {arr.std():.1f}"
              f"   slope = {sl:.3f}")
