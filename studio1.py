import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#A


def consteq(T: float) -> np.ndarray:
    """Equilibrium constants Keq(T) for the two reactions."""
    Keq_ref = np.array([6.428e-6, 2.493e-2], dtype=float)  
    delHr_fit = np.array([-59.032, 38.916], dtype=float) * 1e3 
    R = 8.314
    Tref = 300 + 273.15
    Keq = Keq_ref *np.exp((delHr_fit / R )*(1.0 / Tref - 1.0 / T)) #np.exp?
    return Keq


def r_calc(Pi_bar: np.ndarray, T: float, Keq: np.ndarray) -> np.ndarray: #hittar rateutryck

    PH2, PCO2, PCH3OH, PCO, PH2O = Pi_bar

    Tref = 300 + 273.15
    R = 8.314

    kref = np.array([6.9e-4, 1.8e-3], dtype=float)


    Ea = np.array([35.7, 54.5], dtype=float) * 1e3


    Kref = np.array([0.76, 0.79], dtype=float)  

    
    delHa = np.array([-12.5, -25.9], dtype=float) * 1e3 #ads H = entalpiändring när molekyl adsorberas, högt neg > starkare ads

    k = kref * np.exp(Ea / R * (1.0 / Tref - 1.0 / T))
    K = Kref * np.exp(delHa / R * (1.0 / Tref - 1.0 / T))

    näm = (1.0 + K[1] * PCO2 + np.sqrt(K[0] * PH2)) ** 2

    
    r1 = (k[0] * (PCO2 * PH2 - (PCH3OH * PH2O) /(PH2**2 * Keq[0]))) / näm
    r2 = (k[1]* (PCO2 * PH2**0.5 - (PCO * PH2O) / (Keq[1]*PH2**0.5))) / näm

    return np.array([r1, r2], dtype=float)


def _partials(F: np.ndarray, Ptot_bar: float) -> np.ndarray:
    Ft = np.sum(F)
    y = F / Ft
    return y * Ptot_bar


def pfr_isothermal_rhs(W: float, F: np.ndarray, Ptot_bar: float, T: float) -> np.ndarray:
    """
    Isothermal packed bed reactor:
      dF/dW = nu * r
    where W is catalyst mass [kg cat].
    """
    Pi = _partials(F, Ptot_bar)
    Keq = consteq(T)
    r = r_calc(Pi, T, Keq)  # [r1, r2]

   #stök
    nu = np.array([
        [-3, -1],  # H2
        [-1, -1],  # CO2
        [+1,  0],  # CH3OH
        [ 0, +1],  # CO
        [+1, +1],  # H2O
    ], dtype=float)

    return nu @ r


def run_task_a():
    Ptot_bar = 50.0
    T = 270 + 273.0        # K
    Wtot = 50e3            # kg catalyst

    # Feed: CO2 = 50 mol/s, H2:CO2=3:1
    F0 = np.array([150.0, 50.0, 0.0, 0.0, 0.0], dtype=float)

    sol = solve_ivp(
        fun=lambda W, F: pfr_isothermal_rhs(W, F, Ptot_bar=Ptot_bar, T=T),
        t_span=(0.0, Wtot),
        y0=F0,
        method="BDF",
        atol=1e-10,
        rtol=1e-8
    )

    W = sol.t
    F = sol.y.T  # (n,5)

    X_CO2 = (F0[1] - F[:, 1]) / F0[1]

    # Plots requested in Task A
    plt.figure()
    plt.plot(W, X_CO2)
    plt.xlabel("Catalyst mass W")
    plt.ylabel("CO2 conversion X")
    plt.title("Conversion")

    plt.figure()
    plt.plot(W, F[:, 2])
    plt.xlabel("Catalyst mass W")
    plt.ylabel("F_CH3OH (mol/s)")
    plt.title("Methanol molar flow")

    plt.figure()
    plt.plot(W, F[:, 3])
    plt.xlabel("Catalyst mass W)")
    plt.ylabel("F_CO (mol/s)")
    plt.title("CO molar flow")

    plt.show()

    # Optional: print end values
    print("End flows [H2, CO2, CH3OH, CO, H2O] (mol/s):")
    print(F[-1])
    print("End CO2 conversion:", X_CO2[-1])

    return sol


if __name__ == "__main__":
    run_task_a()