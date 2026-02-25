import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def psat_mmHg(TC, Ant):
    """Antoine: log10(Psat/mmHg) = A - B/(C+T[°C])"""
    A, B, C = Ant
    return 10 ** (A - B / (C + TC))

def find_Tb(TC, x1, Ant_a, Ant_b, P):
    """
    Bubbelpunktsfunktion för binär ideal blandning:
    y1 = x1*Psat1/P, y2 = x2*Psat2/P och villkor y1+y2=1
    """
    Psat1 = psat_mmHg(TC, Ant_a)
    Psat2 = psat_mmHg(TC, Ant_b)
    y1 = x1 * Psat1 / P
    y2 = (1 - x1) * Psat2 / P
    return 1 - y1 - y2


P = 760          # mmHg
q = 0.8          # 80% liquid
zfa = 0.35       # total conc. of A in the inflow (zF)
xfa = 0.3063     # conc. of A in the liquid inflow (given)
xd = 0.9
xw = 0.1
F = 250          # kmol/h
R = 2        # reflux ratio

# Antoine constants
Ant_a = [6.90565, 1211.033, 220.790]
Ant_b = [6.95464, 1344.800, 219.482]


D = F * (zfa - xw) / (xd - xw)
W = F - D

L = R * D
V_rect = D * (R + 1)          # intern ånga i rektifierardelen (över matning)


L_strip = L + q * F
V_strip = V_rect - (1 - q) * F

l = L_strip   # L-streck (stripper)
v = V_strip   # V-streck (stripper)


def y_eq(x):
    return 5 / 3 - 5 / (3 + 4.5 * x)

#
y0 = y_eq(xw)

# x_r: vätskefas på plattan närmast återkokaren (första x i iterationen)
xr = (V_strip / L_strip) * y0 + (W / L_strip) * xw


x = [xr]
y = []
i = 0

while x[i] < xfa:
    y.append(y_eq(x[i]))
    x.append((V_strip / L_strip) * y[i] + (W / L_strip) * xw)
    i = i + 1

m = i + 1  # antal bottnar i stripperdelen (räknat från återkokaren)


while y[i - 1] < xd:
    x.append((V_rect / L) * y[i - 1] + (W * xw - F * zfa) / L)
    i = i + 1
    y.append(y_eq(x[i]))

# Antal steg/bottnar:
total_trays = len(x)    - 1          # fungerar bra med vår lista - återkokare
upper_trays = total_trays - m

TbC = float(fsolve(find_Tb, x0=100.0, args=(xw, Ant_a, Ant_b, P)))
TK = TbC + 273.15  # Kelvin

# Enthalpy: H = a + bT + cT^2  (konstanter givna i uppgiften)
aAg = 0.69381e5
bAg = 0.6752e1
cAg = 0.13199
aBg = 0.31596e5
bBg = 0.15841e2
cBg = 0.15429

HA = aAg + bAg * TK + cAg * TK**2
HB = aBg + bBg * TK + cBg * TK**2

# Ångblandning ut ur reboiler (antar y = y0)
x1_g = y0
x2_g = 1 - y0
Hblandningg = x1_g * HA + x2_g * HB  # [kJ/kmol] (enligt kursfilens enheter)

# Liquid phase enthalpy
aAv = 0.19534e5
bAv = 0.63711e2
cAv = 0.12206
aBv = -0.12588e5
bBv = 0.14150e2
cBv = 0.23130

hA = aAv + bAv * TK + cAv * TK**2
hB = aBv + bBv * TK + cBv * TK**2

# Vätskeblandning i reboiler/understa zonen:
# Vi använder x_r (vätska på första plattan ovan reboiler) – ger korrekt storleksordning.
x1_l = xr
x2_l = 1 - xr
hblandningv = x1_l * hA + x2_l * hB  # [kJ/kmol]

n_dot = F  # kmol/h
Q_reboiler = n_dot * (Hblandningg - hblandningv) / 3600  # kJ/h -> kW

print("=== Resultat ===")
print("Bubbelpunkt reboiler:", TbC, "°C (", TK, "K )")
print("överförd effekt/transfer power:", round(Q_reboiler), "kW")
print("bottnar nedre del/trays lower part:", m)
print("bottnar övre del/trays upper part:", upper_trays)
print("bottnar totalt / trays total:", total_trays)

plt.figure()
plt.plot(range(1, i + 1), y, "r")
plt.plot(range(1, i + 1), x[:i])
plt.legend(["y1", "x1"])
plt.ylabel("x1,y1")
plt.xlabel("Botten nr (Räknat från återkokaren)/ trays nr (counted from the reboiler)")
plt.grid(True)
plt.show()

yfa = y_eq(xfa)
Rmin = (xd - yfa) / (yfa - xfa)
print("Minsta återflödesförhålland/minimum reflux ratio:", Rmin)