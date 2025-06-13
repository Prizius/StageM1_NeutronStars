import numpy as np
from matplotlib import pyplot as plt


## PARAMETERS


pi = np.pi

hbar = 1.054571818e-34  # Js
G = 6.673e-11  # m3/kg/s2
c = 2.9979e8  # m/s
mn = 1.674927500e-27  # kg

Ms = 1.989e30  # kg  mass unit
R = 1000  # m  length unit
eps_0 = 2.0852e25 * 1e9  # 1e9 MeV^4  energy density unit

R_max = 100*R  # Maximum radius, if goes over it, we consider the simulation to diverge
P_lim = 0  # value of P to stop comutation (if need to stop earlier)
dr = 1e-2  # Step size


# Dimentionless factors
A = 4*pi * R**3 * eps_0 / Ms / c / c
C = G * Ms / R / c / c


## LOADING EOS


EOS_DATA_FERMI = np.load("./data/eos_fermi.npz")
EOS_DATA_TEST = np.load("./data/eos_test.npz")
EOS_DATA_FERMI_UR = np.load("./data/eos_fermi_ur.npz")
EOS_DATA_FERMI_CL = np.load("./data/eos_fermi_classical.npz")

EOS_DATA_QCD_NLO_1 = np.load("./data/eos_qcd_nlo_X1.npz")
EOS_DATA_QCD_NLO_2 = np.load("./data/eos_qcd_nlo_X2.npz")
EOS_DATA_QCD_NLO_3 = np.load("./data/eos_qcd_nlo_X3.npz")
EOS_DATA_QCD_NLO_4 = np.load("./data/eos_qcd_nlo_X4.npz")

EOS_DATA_QCD_NNLO_1 = np.load("./data/eos_qcd_nnlo_X1.npz")
EOS_DATA_QCD_NNLO_2 = np.load("./data/eos_qcd_nnlo_X2.npz")
EOS_DATA_QCD_NNLO_3 = np.load("./data/eos_qcd_nnlo_X3.npz")
EOS_DATA_QCD_NNLO_4 = np.load("./data/eos_qcd_nnlo_X4.npz")

EOS_DATA_NJL = np.load("./data/eos_njl.npz")


def eos(P, EOS_DATA):
    """ Computes the interpolated value of epsilon for the given value of P, from the data in EOS_DATA """

    # Prevent extrapolation
    if (P >= np.max(EOS_DATA["P"])).any():
        raise ValueError(f"Il faut des P plus haut : {np.max(EOS_DATA["P"])}")
    if np.logical_and(P > 0, P <= np.min(EOS_DATA["P"])).any():
         raise ValueError(f"Il faut des P plus petits : {np.max(EOS_DATA["P"])}")

    return np.interp(P, EOS_DATA["P"], EOS_DATA["eps"])


def evolution_function(current_states, EOS_DATA):
    r, m, P, _,= current_states.T

    eps = eos(P, EOS_DATA)

    r_evo = 0 * r + 1
    m_evo = A * r**2 * eps
    
    if (m == 0).any():
        P_evo = -A*C * r * P * eps * (1 + P / eps)
    else:
        factor1 = - C * m * eps / r / r
        factor2 = 1 + P / eps
        factor3 = 1 + A * r**3 * P / m
        factor4 = 1 - 2*C * m / r
        P_evo = factor1 * factor2 * factor3 / factor4
    
    new_states = current_states + dr * np.array([r_evo, m_evo, P_evo, 0*r_evo]).T
    new_states[:, 3] = eps
        
    return new_states


def solve_tov(EOS_DATA, Pc):
    N_max = int(R_max/R/dr)

    eps_c = eos(Pc, EOS_DATA)
    initial_states = np.array([0*Pc, 0*Pc, Pc, eps_c]).T  # r, m, P, eps
    
    states = np.zeros((len(Pc), N_max, 4))
    states[:, 0, :] = initial_states

    i = 0
    current_states = initial_states
    while np.any(current_states[:, 2] > P_lim) and i < N_max - 1:
        i += 1
        states[:, i, :] = evolution_function(current_states, EOS_DATA)
        current_states = states[:, i, :]
    
    # Debug
    if i == N_max - 1:
        raise ValueError(f"Ca n'a pas convergé {states[-1, i, 2]}")
    if i < 5:
        raise ValueError("Trop trop trop rapide là")

    return states


def compute_stars_structure(EOS_DATA, Pc, args="", **kwargs):
    res = solve_tov(EOS_DATA, Pc)
    M_list = np.zeros(len(res))
    R_list = np.zeros(len(res))

    for i in range(len(res)):
        index = np.where(res[i, :, 2] < P_lim)[0][0]
        R_list[i] = res[i, index - 1, 0]
        M_list[i] = res[i, index - 1, 1]
    
    plt.plot(R_list, M_list, args, **kwargs)


Pc = 10**np.linspace(-5, 5, 200)



plt.figure()
# compute_stars_structure(EOS_DATA_FERMI, Pc, "k", label="Ideal Fermi gas")
# compute_stars_structure(EOS_DATA_FERMI_CL, Pc, "r", label="Classical limit")
# compute_stars_structure(Pc, label="UR")
# compute_stars_structure(EOS_DATA_QCD_NLO_1, Pc, "r--", label="QCD NLO (X=1)")
# compute_stars_structure(EOS_DATA_QCD_NLO_2, Pc, "g--", label="QCD NLO (X=2)")
# compute_stars_structure(EOS_DATA_QCD_NLO_3, Pc, label="QCD NLO (X=3)")
# compute_stars_structure(EOS_DATA_QCD_NLO_4, Pc, "b--", label="QCD NLO (X=4)")
# compute_stars_structure(EOS_DATA_QCD_NNLO_1, Pc, "r", label="QCD NNLO (X=1)")
# compute_stars_structure(EOS_DATA_QCD_NNLO_2, Pc, "g", label="QCD NNLO (X=2)")
# compute_stars_structure(EOS_DATA_QCD_NNLO_3, Pc, label="QCD NNLO (X=3)")
# compute_stars_structure(EOS_DATA_QCD_NNLO_4, Pc, "b", label="QCD NNLO (X=4)")
compute_stars_structure(EOS_DATA_NJL, Pc, label="NJL")
plt.xlabel("Radius [km]")
plt.ylabel("Mass [$M_\odot$]")


## Making plots prettier

def rot_lim(nuk, m_max=100):
    m = np.zeros(10000)
    r = np.zeros(10000)
    m[:-1] = np.linspace(0, m_max, 9999)
    r[:-1] = 10*(1045/nuk)**(2/3) * m[:-1]**(1/3)
    r[-1] = r[-2]
    return r, m

Rmax = 10
Mmax = 1.5

plt.fill(*rot_lim(641), facecolor="lightblue", edgecolor="none")
plt.fill([0, 3*G*Ms/R/c/c*Mmax*1.3, 0], [0, 1.3*Mmax, 1.3*Mmax], facecolor="lightgray", edgecolor="none")
plt.fill([0, 2*G*Ms/R/c/c*Mmax*1.3, 0], [0, 1.3*Mmax, 1.3*Mmax], facecolor="gray", edgecolor="none")

plt.xlim((0, Rmax*1.2))
plt.ylim((0, Mmax*1.2))

plt.legend()
plt.show()
