# task_b_adiabatic_pfr.py
# Task B: Adiabatic packed-bed reactor (PBR/PFR) solved vs catalyst mass W
# Components order:
#   0:H2, 1:CO2, 2:CH3OH, 3:CO, 4:H2O
#
# Requires: numpy, scipy, matplotlib

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def consteq(T: float) -> np.ndarray:
    Keq_ref = np.array([6.428e-6, 2.493e-2], dtype=float)  
    delHr_fit = np.array([-59.032, 38.916], dtype=float) * 1e3 
    R = 8.314
    Tref = 300 + 273.15
    Keq = Keq_ref *np.exp((delHr_fit / R )*(1.0 / Tref - 1.0 / T)) #np.exp?
    return Keq


def thermo_calc(T: float) -> tuple[np.ndarray, np.ndarray]:
    # cp = A + B*T + C*T^2 + D*T^3  (J/mol/K)

    # H2(g)
    cpA_H2, cpB_H2, cpC_H2, cpD_H2 = 25.399, 2.0178e-2, -3.8549e-5, 3.188e-8
    cp_H2 = cpA_H2 + cpB_H2*T + cpC_H2*T**2 + cpD_H2*T**3

    # CO2(g)
    delHf_CO2 = -393.52  # kJ/mol
    cpA_CO2, cpB_CO2, cpC_CO2, cpD_CO2 = 27.437, 4.2315e-2, -1.9555e-5, 3.9968e-9
    cp_CO2 = cpA_CO2 + cpB_CO2*T + cpC_CO2*T**2 + cpD_CO2*T**3

    # CH3OH(g)
    delHf_CH3OH = -200.7  # kJ/mol
    cpA_CH3OH, cpB_CH3OH, cpC_CH3OH, cpD_CH3OH = 40.046, -3.8287e-2, 2.4529e-4, -2.1679e-7
    cp_CH3OH = cpA_CH3OH + cpB_CH3OH*T + cpC_CH3OH*T**2 + cpD_CH3OH*T**3

    # CO(g)
    delHf_CO = -110.53  # kJ/mol
    cpA_CO, cpB_CO, cpC_CO, cpD_CO = 28.142, 0.167e-2, 0.537e-5, -2.221e-9
    cp_CO = cpA_CO + cpB_CO*T + cpC_CO*T**2 + cpD_CO*T**3

    # H2O(g)
    delHf_H2O = -241.83  # kJ/mol
    cpA_H2O, cpB_H2O, cpC_H2O, cpD_H2O = 33.933, -6.4186e-3, 2.9906e-5, -1.7825e-8
    cp_H2O = cpA_H2O + cpB_H2O*T + cpC_H2O*T**2 + cpD_H2O*T**3

    cpi = np.array([cp_H2, cp_CO2, cp_CH3OH, cp_CO, cp_H2O], dtype=float)

    # reference temperature
    T0 = 298.15

    # Reaction 1: CO2 + 3H2 -> CH3OH + H2O
    delHr0_R1 = (delHf_CH3OH + delHf_H2O - delHf_CO2) * 1e3  # J/mol
    delcpA_R1 = cpA_CH3OH + cpA_H2O - 3*cpA_H2 - cpA_CO2
    delcpB_R1 = cpB_CH3OH + cpB_H2O - 3*cpB_H2 - cpB_CO2
    delcpC_R1 = cpC_CH3OH + cpC_H2O - 3*cpC_H2 - cpC_CO2
    delcpD_R1 = cpD_CH3OH + cpD_H2O - 3*cpD_H2 - cpD_CO2

    delHr_R1 = delHr0_R1 \
               - delcpA_R1*(T - T0) \
               - (delcpB_R1/2)*(T**2 - T0**2) \
               - (delcpC_R1/3)*(T**3 - T0**3) \
               - (delcpD_R1/4)*(T**4 - T0**4)

    # Reaction 2: CO2 + H2 -> CO + H2O
    delHr0_R2 = (delHf_CO + delHf_H2O - delHf_CO2) * 1e3  # J/mol
    delcpA_R2 = cpA_CO + cpA_H2O - cpA_H2 - cpA_CO2
    delcpB_R2 = cpB_CO + cpB_H2O - cpB_H2 - cpB_CO2
    delcpC_R2 = cpC_CO + cpC_H2O - cpC_H2 - cpC_CO2
    delcpD_R2 = cpD_CO + cpD_H2O - cpD_H2 - cpD_CO2

    delHr_R2 = delHr0_R2 \
               - delcpA_R2*(T - T0) \
               - (delcpB_R2/2)*(T**2 - T0**2) \
               - (delcpC_R2/3)*(T**3 - T0**3) \
               - (delcpD_R2/4)*(T**4 - T0**4)

    delHr = np.array([delHr_R1, delHr_R2], dtype=float)
    return delHr, cpi


def r_calc(Pi_bar: np.ndarray, T: float, Keq: np.ndarray) -> np.ndarray:  #Beräkna r
    
    PH2, PCO2, PCH3OH, PCO, PH2O = Pi_bar

    Tref = 300 + 273.15
    R = 8.314

    kref = np.array([6.9e-4, 1.8e-3], dtype=float)
    Ea = np.array([35.7, 54.5], dtype=float) * 1e3
    Kref = np.array([0.76, 0.79], dtype=float)  # [H2, CO2] bar^-1
    delHa = np.array([-12.5, -25.9], dtype=float) * 1e3  # J/mol

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


def pfr_adiabatic_rhs(W: float, U: np.ndarray, Ptot_bar: float) -> np.ndarray:
    F = U[:5].copy()
    T = float(U[5])

    # kinetics inputs
    Pi = _partials(F, Ptot_bar)
    Keq = consteq(T)
    r = r_calc(Pi, T, Keq)  # mol/kg/s

    # stoichiometry
    nu = np.array([
        [-3, -1],  # H2
        [-1, -1],  # CO2
        [+1,  0],  # CH3OH
        [ 0, +1],  # CO
        [+1, +1],  # H2O
    ], dtype=float)

    # mass balances
    dF_dW = nu @ r  # mol/s per kg

    # energy balance (adiabatic)
    delHr, cpi = thermo_calc(T)     # J/mol, J/mol/K
    cp_mix_flow = np.sum(F * cpi)   # J/s/K
    qdot_per_kg = np.sum(r * delHr) # J/kg/s
    dT_dW = - qdot_per_kg / cp_mix_flow

    return np.hstack([dF_dW, dT_dW])


def run_task_b():
    # Given conditions
    Ptot_bar = 50.0
    T0 = 270 + 273.15   # K
    Wtot = 50e3        # kg catalyst

    # Feed: CO2 = 50 mol/s, H2:CO2 = 3:1
    F0 = np.array([150.0, 50.0, 0.0, 0.0, 0.0], dtype=float)
    U0 = np.hstack([F0, T0])

    sol = solve_ivp(
        fun=lambda W, U: pfr_adiabatic_rhs(W, U, Ptot_bar=Ptot_bar),
        t_span=(0.0, Wtot),
        y0=U0,
        method="BDF",
        atol=1e-10,
        rtol=1e-8
    )

    W = sol.t
    U = sol.y.T
    F = U[:, :5]
    T = U[:, 5]

    X_CO2 = (F0[1] - F[:, 1]) / F0[1]


    plt.figure()
    plt.plot(W, X_CO2)
    plt.xlabel("Catalyst mass (kg)")
    plt.ylabel("Conversion")
    plt.title("CO2 conversion (B)")

    plt.figure()
    plt.plot(W, F[:, 1])
    plt.xlabel("Catalyst mass (kg)")
    plt.ylabel("molar flow (mol/s)")
    plt.title("FCO2 (B)")

    plt.figure()
    plt.plot(W, F[:, 2])
    plt.xlabel("Catalyst mass (kg)")
    plt.ylabel("molar flow (mol/s)")
    plt.title("FCH3OH (B)")

    plt.figure()
    plt.plot(W, F[:, 3])
    plt.xlabel("Catalyst mass (kg)")
    plt.ylabel("molar flow (mol/s)")
    plt.title("FCO (B)")

    plt.figure()
    plt.plot(W, T)
    plt.xlabel("Catalyst mass W (kg)")
    plt.ylabel("T (K)")
    plt.title("Temperature(B)")

    plt.show()

    return sol


if __name__ == "__main__":
    run_task_b()
