import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ============================================================
# Data (1 = acetone, 2 = methanol)
# ============================================================
P = 760.0  # mmHg

# Wilson parameters
W12 = 0.65675
W21 = 0.77204

# Antoine constants (T in °C, Psat in mmHg, log10 form)
# Psat = 10^(A - B/(T + C))
A1, B1, C1 = 7.02447, 1161.00, 224.00   # acetone
A2, B2, C2 = 7.87863, 1473.11, 230.00   # methanol


# ============================================================
# Helpers
# ============================================================
def antoine_psat(T, A, B, C):
    return 10.0 ** (A - (B / (T + C)))


def wilson_gammas(x1, W12, W21):
    x1 = float(np.clip(x1, 1e-14, 1 - 1e-14))
    x2 = 1.0 - x1

    ln_gamma1 = -np.log(x1 + W12 * x2) + x2 * (
        (W12 / (x1 + W12 * x2)) - (W21 / (W21 * x1 + x2))
    )
    ln_gamma2 = -np.log(x2 + W21 * x1) - x1 * (
        (W12 / (x1 + W12 * x2)) - (W21 / (W21 * x1 + x2))
    )
    return np.exp(ln_gamma1), np.exp(ln_gamma2)


def K_values(T, x1):
    g1, g2 = wilson_gammas(x1, W12, W21)
    P1sat = antoine_psat(T, A1, B1, C1)
    P2sat = antoine_psat(T, A2, B2, C2)
    K1 = g1 * P1sat / P
    K2 = g2 * P2sat / P
    return K1, K2, g1, g2, P1sat, P2sat


def robust_fsolve(fun, x0_list, tol=1e-10, maxfev=300):
    """
    Prova fsolve med flera startgissningar.
    Returnerar (sol, ier, msg). ier==1 betyder konvergens.
    """
    last = None
    for x0 in x0_list:
        sol, infodict, ier, msg = fsolve(fun, x0=x0, full_output=True, xtol=tol, maxfev=maxfev)
        last = (sol, ier, msg)
        if ier == 1 and np.all(np.isfinite(sol)):
            return sol, ier, msg
    return last


# ============================================================
# Bubble point: given x1, solve sum(y)=1
# y_i = gamma_i x_i Psat_i / P
# ============================================================
def bubble_residual(T, x1):
    x2 = 1.0 - x1
    g1, g2 = wilson_gammas(x1, W12, W21)
    P1sat = antoine_psat(T, A1, B1, C1)
    P2sat = antoine_psat(T, A2, B2, C2)
    y1 = g1 * x1 * P1sat / P
    y2 = g2 * x2 * P2sat / P
    return y1 + y2 - 1.0


def bubble_point_T_and_y(x1, T_guess=60.0):
    # flera startgissningar om det behövs
    x0s = [T_guess, 50.0, 60.0, 70.0]
    sol, ier, msg = robust_fsolve(lambda TT: bubble_residual(TT, x1), x0s)
    if ier != 1:
        raise RuntimeError(f"Bubble fsolve failed for x1={x1:.4g}: {msg}")

    T = float(sol[0]) if np.ndim(sol) else float(sol)

    x2 = 1.0 - x1
    g1, g2 = wilson_gammas(x1, W12, W21)
    P1sat = antoine_psat(T, A1, B1, C1)
    P2sat = antoine_psat(T, A2, B2, C2)

    y1 = g1 * x1 * P1sat / P
    y2 = g2 * x2 * P2sat / P
    ysum = y1 + y2
    return T, float(y1 / ysum), float(y2 / ysum)


# ============================================================
# Dew point: given y1, solve unknowns (T, x1)
# y1 = gamma1(x)*x1*Psat1(T)/P
# y2 = gamma2(x)*x2*Psat2(T)/P
# ============================================================
def dew_equations(X, y1):
    T, x1 = X
    x1 = float(np.clip(x1, 1e-14, 1 - 1e-14))
    x2 = 1.0 - x1
    y2 = 1.0 - y1

    g1, g2 = wilson_gammas(x1, W12, W21)
    P1sat = antoine_psat(T, A1, B1, C1)
    P2sat = antoine_psat(T, A2, B2, C2)

    eq1 = y1 - g1 * x1 * P1sat / P
    eq2 = y2 - g2 * x2 * P2sat / P
    return [eq1, eq2]


def dew_point_T_and_x(y1, T_guess=60.0, x_guess=None):
    if x_guess is None:
        x_guess = y1

    # prova flera startgissningar (ibland behövs det nära 0/1)
    x0s = [
        [T_guess, x_guess],
        [60.0, y1],
        [55.0, min(max(y1, 1e-3), 1 - 1e-3)],
        [65.0, min(max(y1, 1e-3), 1 - 1e-3)],
    ]

    sol, ier, msg = robust_fsolve(lambda X: dew_equations(X, y1), x0s)
    if ier != 1:
        raise RuntimeError(f"Dew fsolve failed for y1={y1:.4g}: {msg}")

    T, x1 = sol
    x1 = float(np.clip(x1, 1e-14, 1 - 1e-14))
    return float(T), x1


# ============================================================
# Flash: given z1 and beta=V/F, solve for (T, x1)
# Using Rachford-Rice + one consistency equation for x1
# ============================================================
def flash_equations(X, z1, beta):
    T, x1 = X
    x1 = float(np.clip(x1, 1e-14, 1 - 1e-14))
    z2 = 1.0 - z1

    K1, K2, *_ = K_values(T, x1)

    RR = (
        z1 * (K1 - 1.0) / (1.0 + beta * (K1 - 1.0))
        + z2 * (K2 - 1.0) / (1.0 + beta * (K2 - 1.0))
    )

    x1_from_z = z1 / (1.0 + beta * (K1 - 1.0))
    eq2 = x1 - x1_from_z
    return [RR, eq2]


def solve_flash(z1=0.5, beta=0.5, T_guess=60.0, x_guess=None):
    if x_guess is None:
        x_guess = z1

    x0s = [
        [T_guess, x_guess],
        [60.0, z1],
        [55.0, z1],
        [65.0, z1],
    ]

    sol, ier, msg = robust_fsolve(lambda X: flash_equations(X, z1, beta), x0s)
    if ier != 1:
        raise RuntimeError(f"Flash fsolve failed: {msg}")

    T, x1 = sol
    x1 = float(np.clip(x1, 1e-14, 1 - 1e-14))
    x2 = 1.0 - x1

    K1, K2, g1, g2, P1sat, P2sat = K_values(T, x1)

    y1 = K1 * x1
    y2 = K2 * x2
    ysum = y1 + y2
    y1 /= ysum
    y2 /= ysum

    alpha = (y1 / x1) / (y2 / x2)

    return {
        "T": float(T),
        "x1": x1,
        "x2": x2,
        "y1": float(y1),
        "y2": float(y2),
        "K1": float(K1),
        "K2": float(K2),
        "gamma1": float(g1),
        "gamma2": float(g2),
        "P1sat": float(P1sat),
        "P2sat": float(P2sat),
        "alpha": float(alpha),
    }


# ============================================================
# Main
# ============================================================
def main():
    # ----- Bubble curve -----
    x1_grid = np.linspace(1e-4, 1 - 1e-4, 101)
    Tb = np.zeros_like(x1_grid)
    y1_bub = np.zeros_like(x1_grid)

    T_guess = 60.0
    for i, x1 in enumerate(x1_grid):
        T, y1, _ = bubble_point_T_and_y(x1, T_guess=T_guess)
        Tb[i] = T
        y1_bub[i] = y1
        T_guess = T  # warm-start

    #Dew curve
    y1_grid = np.linspace(1e-4, 1 - 1e-4, 101)
    Td = np.zeros_like(y1_grid)
    x1_dew = np.zeros_like(y1_grid)

    T_guess = 60.0
    x_guess = 0.5
    for i, y1 in enumerate(y1_grid):
        T, x1 = dew_point_T_and_x(y1, T_guess=T_guess, x_guess=x_guess)
        Td[i] = T
        x1_dew[i] = x1
        T_guess = T
        x_guess = x1

    plt.figure()
    plt.plot(x1_grid, Tb, linewidth=1.8, label="Bubble: T vs x1")
    plt.plot(y1_grid, Td, linewidth=1.8, label="Dew: T vs y1")
    plt.xlabel("x y")
    plt.ylabel("T°C)")
    plt.title("T-x-y")
    plt.grid(True)
    

    plt.figure()
    plt.plot(x1_grid, y1_bub, linewidth=1.8)
    plt.plot([0,1], [0,1], label="y = x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Equilibrium curve")
    plt.grid(True)

    #Flash
    z1 = 0.5
    beta = 0.5
    flash = solve_flash(z1=z1, beta=beta, T_guess=60.0, x_guess=z1)

    print("\n--- FLASH RESULT ---")
    print(f"P      = {P:.1f} mmHg")
    print(f"z1     = {z1:.3f}")
    print(f"beta   = {beta:.3f}  (V/F)")
    print(f"T      = {flash['T']:.4f} °C")
    print(f"x1     = {flash['x1']:.6f}")
    print(f"y1     = {flash['y1']:.6f}")
    print(f"alpha  = {flash['alpha']:.6f}")

    plt.show()


if __name__ == "__main__":
    main()