"""
chapter3_adversarial.py
=======================
Adversarial-context experiment for Chapter 3, §3.4.

Setup: theta* = e_2. Adversarial context switches from e_1 (CMRUP Stage 1)
to e_2 (the signal direction, never seen in Stage 1).

CMRUP's fixed Stage-1 estimate gives theta_hat[1] = 0 -> g_hat = 0
while g_true = 1 -> constant per-round mispricing -> near-linear regret.

VAPE updates theta_hat online via the gatekeeper; after ~400 VA rounds on e_2
the gate drops below mu and PE takes over with an accurate g_hat ~1
-> near-optimal pricing -> sub-linear regret.

PE phase uses greedy UCB (standard practice for empirical demonstrations;
see Sections 3.2-3.3 for the full theoretical PE treatment).

VAPE-NL behaves identically to VAPE in this setting (same gatekeeper VA
phase, uniform noise does not distinguish the two; their separation appears
under non-Lipschitz demand, analysed theoretically in Section 3.4).

Produces:  ch3_adversarial.png
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
d      = 5
Bx     = 1.0
By     = 2.0
Btheta = 1.0
Bxi    = 0.5
lam    = 1.0
MU     = 0.05        # gatekeeper threshold (tuned for practical convergence)
C_PE   = 0.5         # UCB exploration constant in PE phase
N_SEEDS = 15
SEEDS   = list(range(N_SEEDS))

theta_star    = np.zeros(d)
theta_star[1] = 1.0          # CMRUP Stage-1 (all e_1) is blind to this direction

T_ADV = [2_000, 8_000, 32_000, 128_000]

COL = {'vape': '#d6604d', 'cmrup': '#2166ac'}
LAB = {'vape': 'VAPE', 'cmrup': 'CMRUP'}

# ── Noise / demand ─────────────────────────────────────────────────────────────
def sample_noise(rng):
    return rng.uniform(-Bxi, Bxi)

def demand(w):
    return float(np.clip((Bxi - w) / (2.0 * Bxi), 0.0, 1.0))

def oracle_revenue(g):
    w_opt = np.clip((Bxi - g) / 2.0, -Bxi, Bxi)
    p_opt = g + w_opt
    return max(0.0, p_opt * demand(w_opt))

# ── Adversarial context ────────────────────────────────────────────────────────
def ctx(t, T, rng):
    T1 = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    x = np.zeros(d)
    x[0 if t < T1 else 1] = Bx
    return x

# ── Helpers ────────────────────────────────────────────────────────────────────
def vape_eps(T):
    return (d ** 2 * np.log(T) ** 2 / T) ** (1.0 / 3.0)

def cmrup_Delta(T):
    T1 = max(d + 1, int(np.ceil(T ** (2.0 / 3.0))))
    return max(1e-6, 0.08 * By * Bx ** 2 * np.sqrt(d * np.log(T) / T1))

# ── VAPE ───────────────────────────────────────────────────────────────────────
def run_vape(T, seed):
    """
    VA phase: online ridge + gatekeeper (mu = MU_TUNE).
    PE phase: greedy UCB over price grid centered on g_hat.
    No Lipschitz correction applied (tuned for practical T; see §3.2 for
    the full theoretical version with L_xi * eps correction).
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
            # ── VA phase: random price, online ridge update ────────────────
            p_t  = rng.uniform(-By, By)
            xi_t = sample_noise(rng)
            o_t  = float(p_t <= g_t + xi_t)
            ytil = o_t - 0.5          # unbiased for g_t / (2*By)
            Vx   = V_inv @ x_t
            V_inv -= np.outer(Vx, Vx) / (1.0 + float(x_t @ Vx))
            V    += np.outer(x_t, x_t)
            sum_xy    += ytil * x_t
            theta_hat  = 2.0 * By * (V_inv @ sum_xy)
            p_posted   = p_t

        else:
            # ── PE phase: greedy UCB ───────────────────────────────────────
            prices = g_hat + K_arr * eps
            A = np.where((prices >= 0.0) & (prices <= By))[0]
            if len(A) == 0:
                A = np.array([np.argmin(np.abs(prices - np.clip(g_hat, 0.0, By)))])
            p_A = prices[A]
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
    Stage 1: T1 = ceil(T^{2/3}) rounds of random pricing, ridge regression.
    With adversarial contexts (e_1 during Stage 1, e_2 after), theta_hat[1]
    remains ~0. Stage 3 posts prices relative to g_hat = 0 while g_true = 1
    -> constant per-round revenue loss -> near-linear cumulative regret.
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
ALGOS = {'vape': run_vape, 'cmrup': run_cmrup}

def run_all(T_values):
    res   = {a: {T: np.zeros(N_SEEDS) for T in T_values} for a in ALGOS}
    total = len(T_values) * len(ALGOS) * N_SEEDS
    done  = 0
    t0    = time.time()
    for T in T_values:
        for name, fn in ALGOS.items():
            for si, seed in enumerate(SEEDS):
                res[name][T][si] = fn(T, seed)
                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"  {done}/{total}  ({elapsed:.0f}s, ~{eta:.0f}s left)")
    return res

def slope(T_values, means):
    s, *_ = linregress(np.log(np.array(T_values, float)),
                       np.log(np.maximum(np.array(means, float), 1e-12)))
    return s

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Adversarial context: theta*=e2, uniform noise")
    print("=" * 60)
    res = run_all(T_ADV)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for algo in ['vape', 'cmrup']:
        Rmat = np.array([res[algo][T] for T in T_ADV]).T
        mu_r = Rmat.mean(0)
        sig  = Rmat.std(0)
        sl   = slope(T_ADV, mu_r)
        ax.plot(T_ADV, mu_r, 'o-', color=COL[algo], lw=2.2,
                label=f"{LAB[algo]}  (slope = {sl:.2f})")
        ax.fill_between(T_ADV, mu_r - sig, mu_r + sig,
                        color=COL[algo], alpha=0.2)

    # T^{2/3} reference anchored to VAPE
    T_arr   = np.array(T_ADV, float)
    ref_lvl = res['vape'][T_ADV[-1]].mean()
    ax.plot(T_ADV, ref_lvl * (T_arr / T_ADV[-1]) ** (2.0/3.0),
            'k--', lw=0.9, alpha=0.5, label=r'$T^{2/3}$ reference')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Horizon $T$', fontsize=12)
    ax.set_ylabel('Cumulative regret $R_T$', fontsize=12)
    ax.set_title('Adversarial contexts: VAPE adapts, CMRUP fails', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    note = (
        r'$\theta^*=e_2$. Context: $x_t=e_1$ for $t<\lceil T^{2/3}\rceil$ (CMRUP Stage 1),' '\n'
        r'then $x_t=e_2$ — direction CMRUP never observed in Stage 1.' '\n'
        r'Uniform noise $\xi_t\!\sim\!\mathcal{U}[-B_\xi,B_\xi]$, $B_\xi\!=\!0.5$.'
        '\n'
        r'VAPE-NL behaves similarly to VAPE here (§3.3).'
    )
    ax.annotate(note, xy=(0.03, 0.97), xycoords='axes fraction',
                fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ch3_adversarial.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved: ch3_adversarial.png")

    print("\n" + "=" * 60)
    print(f"SUMMARY  (T={T_ADV[-1]:,}, mean ± std, {N_SEEDS} seeds)")
    print("=" * 60)
    for algo in ['vape', 'cmrup']:
        arr  = res[algo][T_ADV[-1]]
        sl   = slope(T_ADV, np.array([res[algo][T] for T in T_ADV]).mean(1))
        print(f"  {LAB[algo]:8s}: R_T = {arr.mean():.1f} ± {arr.std():.1f}"
              f"   slope = {sl:.3f}")
