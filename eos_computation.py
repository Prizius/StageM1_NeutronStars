import numpy as np
from matplotlib import pyplot as plt


pi = np.pi
pi2 = pi**2
mn = 940  # MeV


SAVE = True
def save_eos(title, P, eps):
    """P, eps en MeV^4, sauvegardés en 1e9 MeV^4"""
    if not SAVE:
        print(f"NOT SAVING {title}")
        return

    np.savez(
        f"./data/eos_{title}.npz",
        P = P / 1e9,
        eps = eps / 1e9
    )


## FERMI


xF = np.linspace(0, 13, 2000)


def chi(x):
    return 1/8 * (x * np.sqrt(1 + x**2) * (1 + 2*x**2) - np.log(x + np.sqrt(1 + x**2)))


def phi(x):
    return 3/8 * (x * np.sqrt(1 + x**2) * (2/3*x**2 - 1) + np.log(x + np.sqrt(1 + x**2)))


def compute_eos_fermi():
    P_0 = mn**4/3/pi2
    eps_0 = 3 * P_0

    P_real = P_0 * phi(xF)
    eps_real = eps_0 * chi(xF)

    save_eos("fermi", P_real, eps_real)


def compute_eos_fermi_classical():
    pF = mn * xF

    P_classical = pF**5 / 15/pi2/mn
    eps_classical = mn*pF**3 /3/pi2

    save_eos("fermi_classical", P_classical, eps_classical)


def compute_eos_fermi_ur():
    pF = mn * xF
    
    P_ur = pF**4 / 12 / pi2
    eps_ur = 3*P_ur

    save_eos("fermi_ur", P_ur, eps_ur)



## QCD

Nf = 3
Nc = 3
b0 = (11 - 2*Nf/3)/16/pi2
L0 = 200

def compute_eos_nlo(X):
    mu_min = L0 / X * np.exp(1/4/pi/b0) * 0.99
    mu = mu_min + 10**np.linspace(-2, 4, 5000)
    Pf = Nc*Nf*mu**4/12/pi2
    
    g2 = 1/2/b0/np.log(X*mu/L0)

    P_ratio = (1 - g2/2/pi)
    P_NLO = P_ratio * Pf
    eps_NLO = Pf * (3 * P_ratio + 1/4/pi/b0/(np.log(X*mu/L0)**2))

    save_eos(f"qcd_nlo_X{X}", P_NLO, eps_NLO)


c1 = 0.830189
c2 = 0.505545
d1 = 0.438396
d2 = 1.165107
alpha1 = 1.014939
nu1 = 0.670277
nu2 = 0.899925
nu3 = 0.632526


def compute_eos_nnlo(X):
    mu_min = d2 * X**(-nu2)/3
    mu = (mu_min + 10**np.linspace(-1, 5, 5000))  # en GeV
    Pf = Nc*Nf*mu**4/12/pi2
    
    a = d1*X**nu1
    b = d2 * X**(-nu2)
    c = c1 + c2 * X**nu3
    P_ratio = c - (a * (3*mu)**alpha1) / (3*mu - b)
    P_ratio_deriv = 3 * a * (3*mu)**(alpha1-1) * (alpha1*b - 3*mu*(alpha1 - 1)) / (3*mu - b)**2

    P_NNLO = Pf * P_ratio
    eps_NNLO = Pf * (3 * P_ratio + mu * P_ratio_deriv)

    save_eos(f"qcd_nnlo_X{X}", P_NNLO * 1e12, eps_NNLO * 1e12)  # Passage en MeV^4


## NJL

DATA = [
    [664.3, 2.06 * 2, 5.0, 300, 76.3],
    [587.9, 2.44 * 2, 5.6, 400, 141.4],
    [569.3, 2.81 * 2, 5.5, 500, 234.1],
    [568.6, 3.17 * 2, 5.1, 600, 356.1],
] #  L0,    GL2,      m,   Mvac,B

MU_CRIT = [
    260.09204416171843,
    303.81014324896563,
    337.7247659765772,
    366.14541014910134,
] # MeV

Nf = 2

def F(x):
    return 0.5*np.where(
        x < 1e-8,
        1,
        np.sqrt(1 + x**2) + x**2 / 2 * (np.log(np.sqrt(1 + x**2) - 1) - np.log(np.sqrt(1 + x**2) + 1))
    )


def mass_gap_eq(M, mu, set=1, chiral_limit=False):
    L0, GL2, m0, _, _ = DATA[set]
    if chiral_limit:
        m0 = 0
    pF = np.where(M >= mu, 0, np.sqrt(mu**2 - M**2))

    return m0 + GL2*Nc*Nf/pi2 * M * (F(M/L0) - np.where(pF == 0, 0, (pF/L0)**2 * F(M/pF)))


def compute_mass_gap(mu, set=1, chiral_limit=False):
    """ Using a dichotomy method as scipy did not want to cooperate """
    _, _, m0, Mvac, _ = DATA[set]

    m_down = m0
    m_up = 2*Mvac

    val_down = mass_gap_eq(m_down, mu, set=set, chiral_limit=chiral_limit,) - m_down
    val_up = mass_gap_eq(m_up, mu, set=set, chiral_limit=chiral_limit,) - m_up

    if val_up == 0:
        return m_up
    if val_down == 0:
        return m_up

    while np.abs(val_up - val_down) > 1e-2:
        m_mid = (m_down + m_up)/2
        val_mid = mass_gap_eq(m_mid, mu, set=set, chiral_limit=chiral_limit,) - m_mid
        if val_mid * val_up > 0:
            m_up, val_up = m_mid, val_mid
        else:
            m_down, val_down = m_mid, val_mid
    
    return m_mid


def compute_eos_njl(set=1):
    L0, GL2, m0, Mvac, B = DATA[set]

    mu = 10**np.linspace(2, 5, 5000)
    # mu = np.linspace(0, 1000, 1000)
    mg = np.array([compute_mass_gap(mmu, set) for mmu in mu])

    Mvac = compute_mass_gap(0, set)
    pF = np.where(mu < mg, 0, np.sqrt(mu**2/mg**2 - 1))

    PF = mg**4/3/pi2 * phi(pF/mg)
    P_NJL = Nc*Nf*PF + ((Mvac - m0)**2 - (mg - m0)**2)*L0**2 /2/GL2 + Nc*Nf/pi2 * (mg**4 * chi(L0/mg) - Mvac**4 * chi(L0/Mvac))
    eps_NJL = Nc*Nf*mg**4/pi2 * chi(pF/mg) - ((Mvac - m0)**2 - (mg - m0)**2)*L0**2 /2/GL2 - Nc*Nf/pi2 * (mg**4 * chi(L0/mg) - Mvac**4 * chi(L0/Mvac)) + B/130.2

    save_eos(f"njl", P_NJL, eps_NJL)



## ACTUAL COMPUTING

if __name__ == "__main__":
    # compute_eos_fermi()
    # compute_eos_fermi_classical()
    # compute_eos_fermi_ur()

    # compute_eos_nlo(1)
    # compute_eos_nlo(2)
    # compute_eos_nlo(3)
    # compute_eos_nlo(4)

    # compute_eos_nnlo(1)
    # compute_eos_nnlo(2)
    # compute_eos_nnlo(3)
    # compute_eos_nnlo(4)

    compute_eos_njl(1)