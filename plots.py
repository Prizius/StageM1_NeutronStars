import numpy as np
from matplotlib import pyplot as plt
import scipy
import random as rd
import time
from eos_computation import compute_mass_gap


pi = np.pi
pi2 = pi**2
mn = 940  # MeV


def chi(x):
    return 1/8 * (x * np.sqrt(1 + x**2) * (1 + 2*x**2) - np.log(x + np.sqrt(1 + x**2)))


def phi(x):
    return 3/8 * (x * np.sqrt(1 + x**2) * (2/3*x**2 - 1) + np.log(x + np.sqrt(1 + x**2)))


def compare_ur_cl():
    xF = 10**np.linspace(-1, 1, 1000)
    pF = xF*mn
    P_0 = mn**4/3/pi2
    P_real = P_0 * phi(xF)
    P_classical = pF**5 / 15/pi2/mn
    P_ur = pF**4 / 12 / pi2

    plt.figure()
    plt.semilogx([0.1, 10], [1, 1], "k--")
    plt.semilogx(xF, P_classical/P_real, "r", label="Classical")
    plt.semilogx(xF, P_ur/P_real, "b", label="Ultra relativistic")
    plt.grid(True, which="both")
    plt.xlabel("$x_F$", fontsize=12)
    plt.ylabel("$ \\frac{P}{P_f}$  ", rotation=0, fontsize=18)
    plt.legend()
    plt.show()


def plot_eos_fermi():
    EOS_FERMI = np.load(f"./data/eos_fermi.npz")
    EOS_UR = np.load("./data/eos_fermi_ur.npz")
    EOS_CL = np.load("./data/eos_fermi_classical.npz")

    plt.figure()

    plt.loglog(EOS_FERMI["P"][1:], EOS_FERMI["eps"][1:], "k", label="Ideal Fermi gas of neutron", linewidth=3)
    plt.loglog(EOS_CL["P"][1:], EOS_CL["eps"][1:], "r", label="Classical limit")
    plt.loglog(EOS_UR["P"][1:], EOS_UR["eps"][1:], "b", label="Ultra relativistic limit")

    plt.grid()
    plt.xlabel("$P$   [$10^9$ MeV$^4$]", fontsize=12)
    plt.ylabel("$E$   [$10^9$ MeV$^4$]   ", fontsize=12)
    plt.legend()
    plt.show()
    

def plot_eos_njl():
    # EOS_NJL_0 = np.load(f"./data/eos_njl_set0.npz")
    EOS_NJL_1 = np.load(f"./data/eos_njl_set1.npz")
    # EOS_NJL_2 = np.load(f"./data/eos_njl_set2.npz")
    # EOS_NJL_3 = np.load(f"./data/eos_njl_set3.npz")

    plt.figure()

    # plt.plot(EOS_NJL_0["P"][1:], EOS_NJL_0["eps"][1:], "r", label="Set 1")
    plt.loglog(EOS_NJL_1["P"][1:], EOS_NJL_1["eps"][1:]) #, "g", label="Set 2")
    # plt.plot(EOS_NJL_2["P"][1:], EOS_NJL_2["eps"][1:], "b", label="Set 3")
    # plt.plot(EOS_NJL_3["P"][1:], EOS_NJL_3["eps"][1:], "orange", label="Set 4")

    plt.grid()
    plt.xlabel("$P$   [$10^9$ MeV$^4$]", fontsize=12)
    plt.ylabel("$E$   [$10^9$ MeV$^4$]   ", fontsize=12)
    plt.xlim((5e-5, 5e4))
    plt.ylim((.5, 5e6))
    # plt.legend()
    plt.show()


def plot_eos_nlo():
    EOS_NLO_0 = np.load(f"./data/eos_qcd_nlo_X1.npz")
    EOS_NLO_1 = np.load(f"./data/eos_qcd_nlo_X2.npz")
    # EOS_NLO_2 = np.load(f"./data/eos_qcd_nlo_X3.npz")
    EOS_NLO_3 = np.load(f"./data/eos_qcd_nlo_X4.npz")

    plt.figure()

    plt.loglog(EOS_NLO_0["P"][1:], EOS_NLO_0["eps"][1:], "r", label="X=1")
    plt.loglog(EOS_NLO_1["P"][1:], EOS_NLO_1["eps"][1:], "g", label="X=2")
    # plt.loglog(EOS_NLO_2["P"][1:], EOS_NLO_2["eps"][1:], "b", label="X=3")
    plt.loglog(EOS_NLO_3["P"][1:], EOS_NLO_3["eps"][1:], "orange", label="X=4")

    plt.grid()
    plt.xlabel("$P$   [$10^9$ MeV$^4$]", fontsize=12)
    plt.ylabel("$E$   [$10^9$ MeV$^4$]   ", fontsize=12)
    plt.legend()
    plt.show()


def plot_eos_nnlo():
    EOS_NNLO_0 = np.load(f"./data/eos_qcd_nnlo_X1.npz")
    EOS_NNLO_1 = np.load(f"./data/eos_qcd_nnlo_X2.npz")
    # EOS_NLO_2 = np.load(f"./data/eos_qcd_nlo_X3.npz")
    EOS_NNLO_3 = np.load(f"./data/eos_qcd_nnlo_X4.npz")

    plt.figure()

    plt.loglog(EOS_NNLO_0["P"][1:], EOS_NNLO_0["eps"][1:], "r", label="X=1")
    plt.loglog(EOS_NNLO_1["P"][1:], EOS_NNLO_1["eps"][1:], "g", label="X=2")
    # plt.loglog(EOS_NLO_2["P"][1:], EOS_NLO_2["eps"][1:], "b", label="X=3")
    plt.loglog(EOS_NNLO_3["P"][1:], EOS_NNLO_3["eps"][1:], "orange", label="X=4")

    plt.grid()
    plt.xlabel("$P$   [$10^9$ MeV$^4$]", fontsize=12)
    plt.ylabel("$E$   [$10^9$ MeV$^4$]   ", fontsize=12)
    plt.xlim((1e-6, 1e6))
    plt.ylim((5e-2, 5e6))
    plt.legend()
    plt.show()


def compare_eos_nlo_nnlo():
    EOS_NLO_1 = np.load(f"./data/eos_qcd_nlo_X1.npz")
    EOS_NNLO_1 = np.load(f"./data/eos_qcd_nnlo_X1.npz")
    EOS_NLO_2 = np.load(f"./data/eos_qcd_nlo_X2.npz")
    EOS_NNLO_2 = np.load(f"./data/eos_qcd_nnlo_X2.npz")
    EOS_NLO_3 = np.load(f"./data/eos_qcd_nlo_X3.npz")
    EOS_NNLO_3 = np.load(f"./data/eos_qcd_nnlo_X3.npz")

    plt.figure()

    plt.loglog(EOS_NLO_1["P"][1:], EOS_NLO_1["eps"][1:], "r--")
    plt.loglog(EOS_NNLO_1["P"][1:], EOS_NNLO_1["eps"][1:], "r", label="(X=1)")
    plt.loglog(EOS_NLO_2["P"][1:], EOS_NLO_2["eps"][1:], "g--")
    plt.loglog(EOS_NNLO_2["P"][1:], EOS_NNLO_2["eps"][1:], "g", label="(X=2)")
    plt.loglog(EOS_NLO_3["P"][1:], EOS_NLO_3["eps"][1:], "b--")
    plt.loglog(EOS_NNLO_3["P"][1:], EOS_NNLO_3["eps"][1:], "b", label="(X=3)")

    plt.grid()
    plt.xlabel("$P$   [$10^9$ MeV$^4$]", fontsize=12)
    plt.ylabel("$E$   [$10^9$ MeV$^4$]   ", fontsize=12)
    plt.xlim((1e-6, 1e6))
    plt.ylim((5e-2, 5e6))
    plt.legend()
    plt.show()


def plot_mass_gap():
    mu =  np.linspace(300, 500, 1000)
    mg = np.array([compute_mass_gap(mmu) for mmu in mu])
    mgcl = np.array([compute_mass_gap(mmu, chiral_limit=True) for mmu in mu])

    plt.figure()

    transindex = np.where(mg > 200)[0][-1] + 1
    plt.plot(mu[:transindex], mg[:transindex], "k", label="$m = 5.6$ MeV")
    plt.plot(mu[transindex:], mg[transindex:], "k")
    
    transindex = np.where(mgcl > 200)[0][-1] + 1
    plt.plot(mu[:transindex], mgcl[:transindex], "k--", label="Chiral limit $m = 0$ MeV")
    plt.plot(mu[transindex:], mgcl[transindex:], "k--")

    plt.grid()
    plt.xlabel("$\mu$   [MeV]", fontsize=12)
    plt.ylabel("$M$   [MeV]", fontsize=12)
    plt.legend()
    plt.show()


def plot_asymp_freedom():
    Nf = 3
    b0 = (11 - 2*Nf/3)/16/pi2
    print(b0)
    L0 = 200
    L = np.linspace(L0, 5*L0, 1000)
    g2 = 1/2/b0/np.log(L/L0)

    plt.figure()

    plt.plot(L, g2, "k")

    plt.grid()
    plt.xlabel("$\Lambda$   [MeV]", fontsize=12)
    plt.ylabel("$g^2$", fontsize=12)
    # plt.xlim((1e-6, 1e6))
    # plt.ylim((5e-2, 5e6))
    # plt.legend()
    plt.show()




if __name__ == "__main__":
    # compare_ur_cl()
    # plot_eos_fermi()
    # plot_eos_njl()
    # plot_eos_nlo()
    # plot_eos_nnlo()
    # compare_eos_nlo_nnlo()
    # plot_asymp_freedom()
    plot_mass_gap()
    