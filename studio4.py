import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# ========================
# Data input
# ========================

mc = 4.0          # Massflöde kalla/produktflödet [kg/s]
cpc = 2.4e3       # Värmekapacitet produktflödet [J/(kg K)]
Tc_in = 20.0      # Temperatur in på produktflödet [°C]
Tc_ut = 100.0     # Önskad sluttemp på produktflödet [°C]

mh = 3.0          # Massflöde kondensatflödet (varma sidan) [kg/s]
cph = 4.18e3      # Värmekapacitet kondensatflödet [J/(kg K)]
Th_in = 90.0      # Temperatur in på kondensatet [°C]

Abef = 25.0       # Area av befintlig värmeväxlare [m²]
U = 1500.0        # Värmegenomgångstal [W/(m² K)]

Ka = 600.0        # Årlig kostnad för vvx-yta [SEK/(m² år)]
beta = 0.1        # Värde på sparad ånga [SEK/kWh]
t_operation = 4000.0  # Drifttid [h/år]


# ========================
# Grundläggande storheter
# ========================

# Kapacitetsflöden [W/K]
Cc = mc * cpc           # Kalla sidans kapacitetsflöde
Ch = mh * cph           # Varma sidans kapacitetsflöde
Cmin = min(Cc, Ch)      # Minsta kapacitetsflödet
Cmax = max(Cc, Ch)      # Största kapacitetsflödet
Cr = Cmin / Cmax        # Kapacitetsflödeskvot

dT_max = Th_in - Tc_in  # Maximal tempdifferens [K]


# ========================
# Hjälpfunktioner
# ========================

def epsilon_countercurrent(NTU, Cr):
    """
    Temperaturverkningsgrad ε för motströms-vvx.
    Formeln kommer från ε-NTU-metoden (WWW ekv 22-25).

    All beräkning av överförd effekt bygger på detta ε.
    """
    # Undvik division med 0 vid Cr -> 1
    if np.isclose(Cr, 1.0):
        # Gränsfall: Cr ~ 1, förenklad form
        return NTU / (1.0 + NTU)
    # Allmänt fall
    return (1.0 - np.exp(-NTU * (1.0 - Cr))) / (1.0 - Cr * np.exp(-NTU * (1.0 - Cr)))


def epsilon_of_A(A):
    """
    Beräkna ε som funktion av total area A [m²].

    pss ovan vad den används till.
    """
    NTU = U * A / Cmin  # NTU = UA / Cmin
    return epsilon_countercurrent(NTU, Cr) 


def q_kW(A):
    """
    Överförd värmeeffekt från kondensatet till produkten [kW]
    för given total area A (alla förvärmare tillsammans).
    q = ε * Cmin * (Th_in - Tc_in).

    Formeln beskriver hur mke effekt i kW som värmeväxlaren överför vid ett visst A.

    Nyttjas för att ber. hur mycket av produktens tot värmebehov täcks av kondensatet.
    Även för att räkna ut hur ångbehovet minskar vid ökad area.
    """
    eps = epsilon_of_A(A)
    q_W = eps * Cmin * dT_max        # [W]
    return q_W / 1000.0              # [kW]


# ========================
# Profitfunktion
# ========================

# Effekt som återvinns med endast befintlig vvx
q0_kW = q_kW(Abef)    # [kW]


def profit(A):
    """
    Årlig vinst [SEK/år] som funktion av total area A [m²].
    Extra sparad ånga relativt endast befintlig vvx
    minus årlig kostnad för extra area.
    """
    # Sparad effekt (kW) relativt bara den gamla vvx:en
    q_saved_kW = q_kW(A) - q0_kW

    # Sparad energi per år [kWh/år]
    E_saved_kWh_per_year = q_saved_kW * t_operation

    # Värde av sparad ånga per år [SEK/år]
    value_saved = E_saved_kWh_per_year * beta

    # Extra area jämfört med den befintliga [m²]
    A_new = A - Abef

    # Kostnad för den extra ytan [SEK/år]
    cost_vvx = Ka * A_new

    # Årlig vinst
    return value_saved - cost_vvx


def dprofit_dA(A):
    """
    Numerisk derivata d(profit)/dA med liten steglängd.
    Används tillsammans med fsolve för att hitta optimala arean.
    """
    dA = 1e-4 * max(1.0, A)  # Litet steg relativt A
    return (profit(A + dA) - profit(A - dA)) / (2.0 * dA)


# ========================
# Lösning: optimal area
# ========================

# Startgissning för totala arean (befintlig + ny) [m²]
A_guess = Abef + 20.0

# Lös ekvationen d(profit)/dA = 0
A_opt_total = fsolve(lambda A: dprofit_dA(A[0]), x0=[A_guess])[0]

# Säkerställ att A_opt_total inte blir mindre än befintlig area
A_opt_total = max(A_opt_total, Abef)

# Ny optimal area (den nya vvx:en)
A_new_opt = A_opt_total - Abef


# ========================
# Beräkning av effekter
# ========================

# Total värmeeffekt som produkten behöver för att gå från Tc_in till Tc_ut
Q_total_kW = mc * cpc * (Tc_ut - Tc_in) / 1000.0   # [kW]

# Effekt från kondensatet före optimering (endast befintlig vvx)
q_old_kW = q_kW(Abef)                              # [kW]
steam_old_kW = Q_total_kW - q_old_kW               # Ångbehov före ny vvx [kW]

# Effekt från kondensatet efter optimering (bef + ny vvx)
q_new_kW = q_kW(A_opt_total)                       # [kW]
steam_new_kW = Q_total_kW - q_new_kW               # Ångbehov efter ny vvx [kW]

# Absolut minskning av ångbehov [kW]
Q_saved_kW = steam_old_kW - steam_new_kW

# Relativ minskning av ångbehov [%]
rel_saved_percent = (Q_saved_kW / steam_old_kW) * 100.0


# ========================
# Enkel plott av vinst mot area (frivilligt, pedagogiskt)
# ========================

A_values = np.linspace(Abef, Abef + 40, 200)   # Total area från befintlig till +40 m²
profit_values = [profit(A) for A in A_values]

plt.figure()
plt.plot(A_values, profit_values)  # Standardfärg, inget färgval
plt.axvline(A_opt_total, linestyle='--')  # Markera optimum
plt.xlabel('Total area A [m²]')
plt.ylabel('Årlig vinst [SEK/år]')
plt.title('Ekonomisk optimering av värmeväxlaryta')
plt.grid(True)
plt.show()  # Avkommentera om du vill se grafen


# ========================
# Presentera resultatet
# ========================

print("===== Resultat för Studio 4 – VVX-optimering =====")
print(f"Optimal total area (befintlig + ny): {A_opt_total:.2f} m²")
print(f"Optimal ny area (A_new):             {A_new_opt:.2f} m²")

print("")
print("Ångbehov före ny vvx:")
print(f"  Steam_old: {steam_old_kW:.2f} kW")

print("Ångbehov efter ny vvx:")
print(f"  Steam_new: {steam_new_kW:.2f} kW")

print("")
print("Besparing:")
print(f"  Q_saved (absolut): {Q_saved_kW:.2f} kW")
print(f"  Q_saved (relativ): {rel_saved_percent:.1f} % av ursprungligt ångbehov")