"""
Chapter 3: VAPE vs CMRUP vs VAPE-NL.

Four settings: i.i.d. or adversarial contexts, with uniform (Lipschitz) or
cliff (non-Lipschitz) noise. Mean and standard deviation over 15 seeds.

Outputs: ch3_iid_comparison.png, ch3_adversarial_A.png, ch3_adversarial_B.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
from scipy.stats import linregress

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OUT, exist_ok=True)

# ── Global parameters ─────────────────────────────────────────────────────
d       = 5        # ambient dimension (smaller than Ch.4 for speed)
Bx      = 1.0
By      = 2.0
Btheta  = 1.0
Bxi     = 0.5      # noise bound
lam     = 1.0
N_SEEDS = 15
SEEDS   = list(range(N_SEEDS))

# theta* has non-zero components along e_1 and e_2
theta_star        = np.zeros(d)
theta_star[0]     = 1.0 / np.sqrt(2)
theta_star[1]     = 1.0 / np.sqrt(2)

# Horizons used for the log-log rate plot
T_LIST  = [4_000, 16_000, 64_000, 200_000]
T_ADV   = [2_000, 8_000,  32_000, 128_000]   # adversarial (shorter for speed)

# Algorithm colours (consistent with Ch.4 palette)
COL = {'vape': '#d6604d', 'cmrup': '#2166ac', 'vape_nl': '#4dac26'}
LAB = {'vape': 'VAPE', 'cmrup': 'CMRUP', 'vape_nl': 'VAPE-NL'}

# ── Noise / demand ────────────────────────────────────────────────────────
def sample_noise(noise_type, rng):
    """Draw one noise realisation."""
    if noise_type == 'uniform':
        # xi ~ Uniform[-Bxi, Bxi]  —  Lipschitz CDF, L_xi = 1/(2*Bxi)
        return rng.uniform(-Bxi, Bxi)
    else:
        # cliff: xi in {-Bxi/2, +Bxi/2} with equal probability
        # CDF has jumps of 1/2 at ±Bxi/2  —  non-Lipschitz
        return Bxi / 2.0 * (1 if rng.random() < 0.5 else -1)


def demand(w, noise_type):
    """D(w) = P(xi >= w), the true demand at residual w = price - valuation."""
    if noise_type == 'uniform':
        return float(np.clip((Bxi - w) / (2.0 * Bxi), 0.0, 1.0))
    else:  # cliff
        if w <= -Bxi / 2.0:
            return 1.0
        elif w <= Bxi / 2.0:
            return 0.5
        else:
            return 0.0


def oracle_revenue(g, noise_type):
    """max_{p>=0} p * D(p - g)  evaluated analytically."""
    if noise_type == 'uniform':
        # optimal residual: w* = (Bxi - g)/2, clamped to [-Bxi, Bxi]
        w_opt = np.clip((Bxi - g) / 2.0, -Bxi, Bxi)
        p_opt = g + w_opt
        return max(0.0, p_opt * demand(w_opt, 'uniform'))
    else:
        # cliff: two local optima at w = -Bxi/2 and w = Bxi/2 - epsilon
        r_low  = max(0.0, g - Bxi / 2.0) * 1.0   # demand = 1 below lower jump
        r_high = max(0.0, g + Bxi / 2.0) * 0.5   # demand = 0.5 below upper jump
        return max(r_low, r_high)


# ── Context generators ────────────────────────────────────────────────────
def ctx_iid(t, T, rng):
    """Isotropic contexts on the unit sphere in R^d."""
    x = rng.standard_normal(d)
    return x / np.linalg.norm(x) * Bx


def ctx_adversarial(t, T, rng):
    """
    Deterministic adversarial sequence:
      - rounds  0 .. T1-1  : x_t = e_1 only   (covers CMRUP's Stage 1)
      - rounds  T1 .. T-1  : x_t = e_2 only   (direction unseen in Stage 1)
    This makes (X^T X) from Stage 1 rank-1, so CMRUP's theta_hat has
    zero component along e_2, causing systematic mispricing in Stage 3.
    VAPE / VAPE-NL adapt via the gatekeeper and recover.
    """
    T1 = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    x = np.zeros(d)
    x[0 if t < T1 else 1] = Bx
    return x


# ── Shared parameter helpers ──────────────────────────────────────────────
def vape_eps(T):
    return (d ** 2 * np.log(T) ** 2 / T) ** (1.0 / 3.0)


MU_TUNE = 0.05   # practical tuning: theoretical formula gives mu~0.002, far too small
                  # for PE phase to start at T<=200k. Tuned constant matches the
                  # paper's practice of setting mu via cross-validation.

def vape_mu(T, eps):
    return MU_TUNE


def cmrup_Delta(T):
    """
    Grid step for CMRUP, derived from Stage-1 estimation error bound.
    Tuned with multiplicative constant c_tune = 0.08 following the
    paper's practice of tuning rather than using theoretical constants.
    """
    T1    = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    c_tune = 0.08
    return c_tune * By * Bx ** 2 * np.sqrt(d * np.log(T) / T1)


# ── Algorithm 1: VAPE ─────────────────────────────────────────────────────
def run_vape(T, ctx_fn, noise_type, seed):
    """
    VAPE (Algorithm 1 from Chapter 1/3).
    VA phase: gatekeeper + ridge estimator.
    PE phase: successive elimination with L_xi * eps bias correction.
      - uniform noise: L_xi = 1/(2*Bxi)  (correct)
      - cliff   noise: L_xi = 0           (wrong — cannot bound jump bias)
    """
    rng  = np.random.default_rng(seed)
    eps  = vape_eps(T)
    mu   = vape_mu(T, eps)
    alph = T ** (-4)
    Lxi  = 1.0 / (2.0 * Bxi) if noise_type == 'uniform' else 0.0

    # Increment grid: k in {-K_max .. K_max}, price = g_hat + k*eps
    K_max = int(np.ceil((By + Btheta) / eps)) + 2
    K_arr = np.arange(-K_max, K_max + 1)
    n_arms = len(K_arr)

    # Ridge VA state
    V      = lam * np.eye(d)
    V_inv  = np.eye(d) / lam
    sum_xy = np.zeros(d)
    theta_hat = np.zeros(d)

    # PE state
    N_pe = np.zeros(n_arms, dtype=np.int64)
    D_pe = np.zeros(n_arms)

    R = 0.0
    for t in range(T):
        x_t   = ctx_fn(t, T, rng)
        g_t   = x_t @ theta_star   # true valuation (oracle)
        g_hat = x_t @ theta_hat    # algorithm's estimate

        xVx  = float(x_t @ (V_inv @ x_t))
        gate = np.sqrt(max(xVx, 0.0))

        if gate > mu:
            # ── VA phase ──────────────────────────────────────────────────
            p_t  = rng.uniform(-By, By)
            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_t <= g_t + xi_t)
            ytil = o_t - 0.5           # unbiased for g_t/(2*By); 2*By applied below
            Vx   = V_inv @ x_t
            denom = 1.0 + float(x_t @ Vx)
            V_inv -= np.outer(Vx, Vx) / denom
            V     += np.outer(x_t, x_t)
            sum_xy += ytil * x_t
            theta_hat = 2.0 * By * (V_inv @ sum_xy)  # tex: 2By·V⁻¹·Σ(os-½)xs
            p_posted = p_t
        else:
            # ── PE phase (successive elimination) ─────────────────────────
            prices  = g_hat + K_arr * eps
            A       = np.where((prices >= 0.0) & (prices <= By))[0]
            if len(A) == 0:
                A = np.array([np.argmin(np.abs(prices - np.clip(g_hat, 0.0, By)))])

            N_A  = N_pe[A]; D_A = D_pe[A]; p_A = prices[A]
            cb   = np.sqrt(2.0 * np.log(1.0 / max(alph, 1e-300)) /
                           np.maximum(N_A, 1))
            UCB  = p_A * np.minimum(1.0, D_A + cb + 2.0 * Lxi * eps)
            LCB  = p_A * np.maximum(0.0, D_A - cb - 2.0 * Lxi * eps)
            best_lcb   = float(np.max(LCB))
            surviving  = A[UCB >= best_lcb]
            if len(surviving) == 0:
                surviving = A[[np.argmax(UCB)]]

            k_sel   = surviving[np.argmin(N_pe[surviving])]
            p_posted = float(np.clip(prices[k_sel], 0.0, By))

            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_pe[k_sel] += 1
            D_pe[k_sel] += (o_t - D_pe[k_sel]) / N_pe[k_sel]

        w_t = p_posted - g_t
        R  += oracle_revenue(g_t, noise_type) - p_posted * demand(w_t, noise_type)

    return R


# ── Algorithm 2: CMRUP ────────────────────────────────────────────────────
def run_cmrup(T, ctx_fn, noise_type, seed):
    """
    CMRUP (Algorithm 2, Xu & Wang, as described in Chapter 3).
    Stage 1: T1 = ceil(T^{2/3}) rounds of uniform random pricing to
             estimate theta* via ridge regression.
    Stage 2: T_w = ceil(T^{2/3}) warmup rounds probing each grid index.
    Stage 3: UCB + redirect pricing using fixed theta_hat from Stage 1.
    The fixed Stage-1 estimator is the only place CMRUP uses i.i.d. contexts.
    """
    rng = np.random.default_rng(seed)
    T1  = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    Tw  = int(np.ceil(T ** (2.0 / 3.0)))
    Delta = max(1e-6, cmrup_Delta(T))

    # Grid: w_grid[j] = -Bxi + 2j*Delta, j = 0..M_c (breakpoints)
    M_c    = max(1, int(np.ceil(Bxi / Delta)))
    w_grid = np.array([-Bxi + 2.0 * j * Delta for j in range(M_c + 1)])

    # PE state
    N_c = np.zeros(M_c, dtype=np.int64)
    D_c = np.zeros(M_c)

    theta_hat = np.zeros(d)
    X_buf = np.zeros((T1, d))
    z_buf = np.zeros(T1)

    R = 0.0
    C_ucb = 0.5   # tuning constant for UCB radius

    for t in range(T):
        x_t = ctx_fn(t, T, rng)
        g_t = x_t @ theta_star   # true valuation

        # ── Stage 1: random exploration ──────────────────────────────────
        if t < T1:
            p_t  = rng.uniform(-By, By)
            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_t <= g_t + xi_t)
            X_buf[t] = x_t
            z_buf[t] = 2.0 * By * (o_t - 0.5)   # unbiased proxy for x_t·theta*
            if t == T1 - 1:
                # Fit ridge estimator once Stage 1 is complete
                XtX = X_buf.T @ X_buf + lam * np.eye(d)
                theta_hat = np.linalg.solve(XtX, X_buf.T @ z_buf)
            p_posted = p_t

        # ── Stage 2: warmup — probe each bracket once in round-robin ─────
        elif t < T1 + Tw:
            g_hat = x_t @ theta_hat
            j     = (t - T1) % M_c
            p_posted = float(np.clip(g_hat + w_grid[j + 1] - 3.0 * Delta,
                                     0.0, By))
            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_c[j] += 1
            D_c[j] += (o_t - D_c[j]) / N_c[j]

        # ── Stage 3: UCB + redirect ───────────────────────────────────────
        else:
            g_hat = x_t @ theta_hat  # theta_hat is FIXED from Stage 1
            b     = C_ucb * np.sqrt(np.log(T) / np.maximum(N_c, 1))
            UCB   = (g_hat + w_grid[1:] + Delta) * np.minimum(1.0, D_c + b)
            j_t   = int(np.argmax(UCB))

            if b[j_t] > Delta:
                a_t = j_t           # direct probe
            elif j_t >= 1:
                a_t = j_t - 1      # redirect to lower bracket
            else:
                a_t = 0            # boundary

            p_posted = float(np.clip(g_hat + w_grid[a_t + 1] - 3.0 * Delta,
                                     0.0, By))
            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_c[a_t] += 1
            D_c[a_t] += (o_t - D_c[a_t]) / N_c[a_t]

        w_t = p_posted - g_t
        R  += oracle_revenue(g_t, noise_type) - p_posted * demand(w_t, noise_type)

    return R


# ── Algorithm 3: VAPE-NL ─────────────────────────────────────────────────
def run_vape_nl(T, ctx_fn, noise_type, seed):
    """
    VAPE-NL (Algorithm 3 from Chapter 3).
    VA phase: identical to VAPE — gatekeeper + online ridge estimator.
             Needs neither i.i.d. contexts nor Lipschitz demand here.
    PE phase: CMRUP's redirect-UCB mechanism with grid w_k = -Bxi + 2k*eps.
             No L_xi correction needed — handles any bounded noise.
    """
    rng  = np.random.default_rng(seed)
    eps  = vape_eps(T)
    mu   = vape_mu(T, eps)
    alph = T ** (-4)

    # NL grid: w_k = -Bxi + 2k*eps, k in {0..M_nl-1}
    M_nl   = max(1, int(np.ceil(Bxi / eps)))
    w_nl   = np.array([-Bxi + 2.0 * k * eps for k in range(M_nl + 1)])

    # VA ridge state
    V      = lam * np.eye(d)
    V_inv  = np.eye(d) / lam
    sum_xy = np.zeros(d)
    theta_hat = np.zeros(d)

    # PE state  (initialise UCB to +inf so every bracket explored once first)
    N_nl = np.zeros(M_nl, dtype=np.int64)
    D_nl = np.zeros(M_nl)

    R = 0.0
    for t in range(T):
        x_t   = ctx_fn(t, T, rng)
        g_t   = x_t @ theta_star
        g_hat = x_t @ theta_hat

        xVx  = float(x_t @ (V_inv @ x_t))
        gate = np.sqrt(max(xVx, 0.0))

        if gate > mu:
            # ── VA phase (unchanged from VAPE) ────────────────────────────
            p_t  = rng.uniform(-By, By)
            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_t <= g_t + xi_t)
            ytil = o_t - 0.5           # unbiased for g_t/(2*By); 2*By applied below
            Vx   = V_inv @ x_t
            denom = 1.0 + float(x_t @ Vx)
            V_inv -= np.outer(Vx, Vx) / denom
            V     += np.outer(x_t, x_t)
            sum_xy += ytil * x_t
            theta_hat = 2.0 * By * (V_inv @ sum_xy)  # tex: 2By·V⁻¹·Σ(os-½)xs
            p_posted = p_t
        else:
            # ── PE phase (redirect-UCB, no L_xi) ─────────────────────────
            b = np.sqrt(np.log(2.0 / max(alph, 1e-300)) /
                        (2.0 * np.maximum(N_nl, 1)))
            # +inf for any bracket never probed yet (optimistic initialisation)
            UCB = np.where(N_nl == 0,
                           np.inf,
                           (g_hat + w_nl[1:] + eps) * np.minimum(1.0, D_nl + b))

            k_t = int(np.argmax(UCB))

            if b[k_t] > eps:
                a_t = k_t        # direct probe
            elif k_t >= 1:
                a_t = k_t - 1   # redirect
            else:
                a_t = 0         # boundary

            p_posted = float(np.clip(g_hat + w_nl[a_t + 1] - 3.0 * eps,
                                     0.0, By))
            xi_t = sample_noise(noise_type, rng)
            o_t  = float(p_posted <= g_t + xi_t)
            N_nl[a_t] += 1
            D_nl[a_t] += (o_t - D_nl[a_t]) / N_nl[a_t]

        w_t = p_posted - g_t
        R  += oracle_revenue(g_t, noise_type) - p_posted * demand(w_t, noise_type)

    return R


# ── Experiment runner ─────────────────────────────────────────────────────
ALGOS = {
    'vape':    run_vape,
    'cmrup':   run_cmrup,
    'vape_nl': run_vape_nl,
}


def run_experiment(T_values, ctx_fn, noise_type, label=''):
    """
    For each T in T_values, run all three algorithms over N_SEEDS seeds.
    Returns dict: results[algo][T] = array of R_T values (shape: N_SEEDS)
    """
    results = {a: {T: np.zeros(N_SEEDS) for T in T_values} for a in ALGOS}
    total = len(T_values) * len(ALGOS) * N_SEEDS
    done  = 0
    t0    = time.time()
    for T in T_values:
        for algo_name, algo_fn in ALGOS.items():
            for si, seed in enumerate(SEEDS):
                results[algo_name][T][si] = algo_fn(T, ctx_fn, noise_type, seed)
                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"  [{label}] {done}/{total}  "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    return results


# ── Plotting helpers ──────────────────────────────────────────────────────
def fitted_slope(T_values, means):
    """Log-log linear regression slope."""
    log_T = np.log(np.array(T_values, dtype=float))
    log_R = np.log(np.maximum(np.array(means, dtype=float), 1e-12))
    slope, *_ = linregress(log_T, log_R)
    return slope


def plot_panel(ax, T_values, results, title, noise_label):
    """Draw one panel (one noise setting) on ax."""
    for algo in ['vape', 'cmrup', 'vape_nl']:
        Rmat = np.array([results[algo][T] for T in T_values]).T  # (N_SEEDS, len(T))
        mu   = Rmat.mean(0)
        sig  = Rmat.std(0)
        slope = fitted_slope(np.array(T_values, float), mu)
        lbl   = f"{LAB[algo]} (slope={slope:.2f})"
        ax.plot(T_values, mu, 'o-', color=COL[algo], lw=2, label=lbl)
        ax.fill_between(T_values, mu - sig, mu + sig,
                        color=COL[algo], alpha=0.18)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Horizon $T$', fontsize=11)
    ax.set_ylabel('Cumulative regret $R_T$', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    # Dashed reference line T^{2/3}
    T_arr = np.array(T_values, float)
    ref = T_arr[-1] ** (-2.0/3.0) * T_arr ** (2.0/3.0) * float(
        np.array([results[a][T_values[-1]] for a in ALGOS]).mean())
    ax.plot(T_values, ref, 'k--', lw=0.8, alpha=0.5, label='$T^{2/3}$ ref')
    # Noise annotation
    ax.annotate(noise_label, xy=(0.03, 0.97), xycoords='axes fraction',
                fontsize=7.5, va='top', color='#444444',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── 1. i.i.d. comparison (uniform and cliff noise) ───────────────────
    print("=" * 60)
    print("Running i.i.d. experiments (uniform noise) ...")
    res_iid_unif = run_experiment(
        T_LIST, ctx_iid, 'uniform', label='iid-uniform')

    print("Running i.i.d. experiments (cliff noise) ...")
    res_iid_cliff = run_experiment(
        T_LIST, ctx_iid, 'cliff', label='iid-cliff')

    # ── Figure 1: updated i.i.d. comparison ─────────────────────────────
    UNIF_LABEL = (r'Uniform noise: $\xi_t \sim \mathcal{U}[-B_\xi, B_\xi]$, '
                  r'$B_\xi=0.5$' '\n'
                  r'(Lipschitz $F$, $L_\xi = 1/(2B_\xi) = 1$)')
    CLIFF_LABEL = (r'Cliff noise: $\xi_t \in \{-B_\xi/2,\,+B_\xi/2\}$ w.p. 1/2 each, '
                   r'$B_\xi=0.5$' '\n'
                   r'(step-function $F$ with jumps at $\pm B_\xi/2$ — non-Lipschitz)')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_panel(axes[0], T_LIST, res_iid_unif,
               'Uniform noise (Lipschitz)', UNIF_LABEL)
    plot_panel(axes[1], T_LIST, res_iid_cliff,
               'Cliff noise (non-Lipschitz)', CLIFF_LABEL)
    fig.suptitle(
        'Cumulative regret: VAPE vs CMRUP vs VAPE-NL, i.i.d. contexts\n'
        rf'Mean $\pm 1$ std over {N_SEEDS} seeds, '
        r'$d=5$, $\theta^*=(e_1+e_2)/\sqrt{2}$, '
        r'$T\in\{4\text{k},16\text{k},64\text{k},200\text{k}\}$',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ch3_iid_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: ch3_iid_comparison.png")

    # ── 2. Adversarial comparison ─────────────────────────────────────────
    print("Running adversarial Setting A (uniform noise) ...")
    res_adv_unif = run_experiment(
        T_ADV, ctx_adversarial, 'uniform', label='adv-uniform')

    print("Running adversarial Setting B (cliff noise) ...")
    res_adv_cliff = run_experiment(
        T_ADV, ctx_adversarial, 'cliff', label='adv-cliff')

    ADV_CTX_LABEL = (
        r'Adversarial contexts: $x_t = e_1$ for $t < T^{2/3}$ (CMRUP Stage 1),' '\n'
        r'then $x_t = e_2$ — direction unseen during Stage 1.')

    # ── Figure 2: Setting A ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_panel(ax, T_ADV, res_adv_unif,
               'Setting A: adversarial contexts + uniform (Lipschitz) noise',
               ADV_CTX_LABEL + '\n' + UNIF_LABEL)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ch3_adversarial_A.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: ch3_adversarial_A.png")

    # ── Figure 3: Setting B ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_panel(ax, T_ADV, res_adv_cliff,
               'Setting B: adversarial contexts + cliff (non-Lipschitz) noise',
               ADV_CTX_LABEL + '\n' + CLIFF_LABEL)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ch3_adversarial_B.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: ch3_adversarial_B.png")

    # ── Numerical summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NUMERICAL SUMMARY  (R_T at largest T, mean ± std)")
    print("=" * 60)
    for label, results, T_vals in [
        ('i.i.d. + uniform',   res_iid_unif,  T_LIST),
        ('i.i.d. + cliff',     res_iid_cliff, T_LIST),
        ('adv. + uniform (A)', res_adv_unif,  T_ADV),
        ('adv. + cliff   (B)', res_adv_cliff, T_ADV),
    ]:
        T_big = T_vals[-1]
        print(f"\n  Setting: {label}  (T={T_big:,})")
        for algo in ['vape', 'cmrup', 'vape_nl']:
            arr = results[algo][T_big]
            slope = fitted_slope(np.array(T_vals, float),
                                 np.array([results[algo][T] for T in T_vals]).mean(1))
            print(f"    {LAB[algo]:10s}: R_T = {arr.mean():.1f} ± {arr.std():.1f}"
                  f"   fitted slope = {slope:.3f}")

    print("\nAll figures saved to:", OUT)
