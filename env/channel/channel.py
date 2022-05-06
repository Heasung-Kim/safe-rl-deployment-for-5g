import math
import numpy as np
from numpy import linalg as LA

# =====================================================================================================================
# Please note that ALL SOURCE CODE used in this file is borrowed from Dr.Faris's project.
# https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks
#
# Mismar, Faris B., Brian L. Evans, and Ahmed Alkhateeb.
# "Deep reinforcement learning for 5G networks: Joint beamforming, power control, and interference coordination."
# IEEE Transactions on Communications 68.3 (2019): 1581-1592.
# =====================================================================================================================


def compute_bf_vector(theta, f_c, M_ULA):


    c = 3e8  # speed of light
    wavelength = c / f_c

    d = wavelength / 2.  # antenna spacing
    k = 2. * math.pi / wavelength

    exponent = 1j * k * d * math.cos(theta) * np.arange(M_ULA)

    f = 1. / math.sqrt(M_ULA) * np.exp(exponent)

    # Test the norm square... is it equal to unity? YES.
    #    norm_f_sq = LA.norm(f, ord=2) ** 2
    #   print(norm_f_sq)

    return f


def compute_channel(x_ue, y_ue, x_bs, y_bs, f_c, M_ULA, prob_LOS):
    # Np is the number of paths p
    PLE_L = 2
    PLE_N = 4
    G_ant = 3 # dBi for beamforming mmWave antennas

    Np = 4

    # theta is the steering angle.  Sampled iid from unif(0,pi).
    theta = np.random.uniform(low=0, high=math.pi, size=Np)

    is_mmWave = (f_c > 25e9)

    if is_mmWave:
        path_loss_LOS = 10 ** (_path_loss_mmWave(x_ue, y_ue, PLE_L, x_bs, y_bs, f_c=f_c, M_ULA=M_ULA) / 10.)
        path_loss_NLOS = 10 ** (_path_loss_mmWave(x_ue, y_ue, PLE_N, x_bs, y_bs, f_c=f_c, M_ULA=M_ULA) / 10.)
    else:
        path_loss_LOS = 10 ** (_path_loss_sub6(x_ue, y_ue, x_bs, y_bs) / 10.)
        path_loss_NLOS = 10 ** (_path_loss_sub6(x_ue, y_ue, x_bs, y_bs) / 10.)

    # Bernoulli for p
    alpha = np.zeros(Np, dtype=complex)
    p = np.random.binomial(1, prob_LOS)

    if (p == 1):
        Np = 1
        alpha[0] = 1. / math.sqrt(path_loss_LOS)
    else:
        ## just changed alpha to be complex in the case of NLOS
        alpha = (np.random.normal(size=Np) + 1j * np.random.normal(size=Np)) / math.sqrt(path_loss_NLOS)

    rho = 1. * 10 ** (G_ant / 10.)

    # initialize the channel as a complex variable.
    h = np.zeros(M_ULA, dtype=complex)

    for p in np.arange(Np):
        a_theta = compute_bf_vector(theta[p], f_c=f_c, M_ULA=M_ULA)
        h += alpha[p] / rho * a_theta.T # scalar multiplication into a vector

    h *= math.sqrt(M_ULA)

    #print ('Warning: channel gain is {} dB.'.format(10*np.log10(LA.norm(h, ord=2))))
    return h

# https://ieeexplore-ieee-org.ezproxy.lib.utexas.edu/stamp/stamp.jsp?tp=&arnumber=7522613
def _path_loss_mmWave(x, y, PLE, x_bs, y_bs, f_c, M_ULA):
    # These are the parameters for f = 28000 MHz.
    c = 3e8  # speed of light
    wavelength = c / f_c
    A = 0.0671
    Nr = M_ULA
    sigma_sf = 9.1
    # PLE = 3.812

    d = math.sqrt((x - x_bs) ** 2 + (y - y_bs) ** 2)  # in meters

    fspl = 10 * np.log10(((4 * math.pi * d) / wavelength) ** 2)
    pl = fspl + 10 * np.log10(d ** PLE) * (1 - A * np.log2(Nr))

    chi_sigma = np.random.normal(0, sigma_sf)  # log-normal shadowing
    L = pl + chi_sigma

    return L  # in dB

def _path_loss_sub6(self, x, y, x_bs=0, y_bs=0):
    f_c = self.f_c
    c = 3e8  # speed of light
    d = math.sqrt((x - x_bs) ** 2 + (y - y_bs) ** 2)
    h_B = 20
    h_R = 1.5

    #        print('Distance from cell site is: {} km'.format(d/1000.))
    # FSPL
    L_fspl = -10 * np.log10((4. * math.pi * c / f_c / d) ** 2)

    # COST231
    C = 3
    a = (1.1 * np.log10(f_c / 1e6) - 0.7) * h_R - (1.56 * np.log10(f_c / 1e6) - 0.8)
    L_cost231 = 46.3 + 33.9 * np.log10(f_c / 1e6) + 13.82 * np.log10(h_B) - a + (
                44.9 - 6.55 * np.log10(h_B)) * np.log10(d / 1000.) + C

    L = L_cost231

    return L  # in dB
