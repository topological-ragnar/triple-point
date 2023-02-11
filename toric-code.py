# taken from https://tenpy.readthedocs.io/en/latest/notebooks/11_toric_code.html

import numpy as np
import scipy
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=120)
plt.rcParams['figure.dpi'] = 150

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.networks.mps import MPS
from tenpy.networks.terms import TermList
from tenpy.models.toric_code import ToricCode, DualSquare
from tenpy.models.lattice import Square

tenpy.tools.misc.setup_logging(to_stdout="WARNING")  # don't show info text


# this version of toric code has explicit wilson and thooft lines added to the ham
class ExtendedToricCode(ToricCode):

    def init_terms(self, model_params):
        ToricCode.init_terms(self, model_params)  # add terms of the original ToricCode model

        Ly = self.lat.shape[1]
        J_WL = model_params.get('J_WL', 0.)
        J_HL = model_params.get('J_HL', 0.)
        # unit-cell indices:
        # u=0: vertical links
        # u=1: horizontal links

        # Wilson Loop
        x, u = 0, 0 # vertical links
        self.add_local_term(-J_WL, [('Sigmaz', [x, y, u]) for y in range(Ly)])

        # t'Hooft Loop
        x, u = 0, 1 # horizontal links
        self.add_local_term(-J_HL, [('Sigmax', [x, y, u]) for y in range(Ly)])

        h = model_params.get('h', 0.)
        for u in range(2):
            self.add_onsite(-h, u, 'Sigmaz')

        g = model_params.get('g', 0.)
        for u in range(2):
            self.add_onsite(-g, u, 'Sigmax')

        k = model_params.get('k', 0.)
        for u in range(2):
            self.add_onsite(-k, u, 'Sigmay')

    def wilson_loop_y(self, psi):
        """Measure wilson loop around the cylinder."""
        Ly = self.lat.shape[1]
        x, u = 0, 0 # vertical links
        W = TermList.from_lattice_locations(self.lat, [[("Sigmaz",[x, y, u]) for y in range(Ly)]])
        return psi.expectation_value_terms_sum(W)[0]

    def hooft_loop_y(self, psi):
        """Measure t'Hooft loop around the cylinder."""
        Ly = self.lat.shape[1]
        x, u = 0, 1 # horizontal links
        H = TermList.from_lattice_locations(self.lat, [[("Sigmax",[x, y, u]) for y in range(Ly)]])
        return psi.expectation_value_terms_sum(H)[0]

dmrg_params = {
    'mixer': True,
    'trunc_params': {'chi_max': 250,
                     'svd_min': 1.e-8},
    'max_E_err': 1.e-8,
    'max_S_err': 1.e-7,
    'N_sweeps_check': 4,
    'max_sweeps':24,
}
model_params = {
    'Lx': 1, 'Ly': 7, # Ly is set below
    'bc_MPS': "infinite",
    'conserve': None,
    'Jv' : 1.0,
    'Jp' : 1.0,
    'J_WL' : 0,
    'J_HL' : 0
}


def run_DMRG(Ly, J_WL, J_HL, h=0.,g=0.,k=0.):
    print("="*80)
    print(f"Start iDMRG for Ly={Ly:d}, J_WL={J_WL:.2f}, J_HL={J_HL:.2f}, h={h:.2f}, g={g:.3f},  k={k:.2f}")
    model_params_clean = model_params.copy()
    model_params_clean['Ly'] = Ly
    model_params_clean['h'] = h
    model_params_clean['g'] = g
    model_params_clean['k'] = k
    model_clean = ExtendedToricCode(model_params_clean)
    model_params_seed = model_params_clean.copy()
    # model_params_seed['J_WL'] = J_WL
    # model_params_seed['J_HL'] = J_HL
    model = ExtendedToricCode(model_params_seed)
    # psi = MPS.from_lat_product_state(model.lat, [[["up"]]])
    B_list = [np.ones((2,1,1)) for i in range(2*Ly)]
    psi = MPS.from_Bflat(model.lat.mps_sites(),B_list,bc='infinite')
    psi.canonical_form()

    eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
    # E0, psi = eng.run()
    # WL = model.wilson_loop_y(psi)
    # HL = model.hooft_loop_y(psi)
    # print(f"after first DMRG run: <psi|W|psi> = {WL: .3f}")
    # print(f"after first DMRG run: <psi|H|psi> = {HL: .3f}")
    # print(f"after first DMRG run: E (including W/H loops) = {E0:.10f}")

    # E0_clean = model_clean.H_MPO.expectation_value(psi)
    # print(f"after first DMRG run: E (excluding W/H loops) = {E0_clean:.10f}")

    # switch to model without Wilson/Hooft loops
    eng.init_env(model=model_clean)

    E1, psi = eng.run()

    WL = model_clean.wilson_loop_y(psi)
    HL = model_clean.hooft_loop_y(psi)
    print(f"after second DMRG run: <psi|W|psi> = {WL: .3f}")
    print(f"after second DMRG run: <psi|H|psi> = {HL: .3f}")
    print(f"after second DMRG run: E (excluding W/H loops) = {E1:.10f}")
    print("max chi: ", max(psi.chi))

    return {'psi': psi,
            'model': model_clean,
            # 'E0': E0, 'E0_clean': E0_clean,
            'E': E1,
            'WL': WL, 'HL': HL,
            'Ly': Ly}
    print("="*80)

def Stopo(results):
    x = np.array([res['Ly'] for res in results.values()])
    A = np.vstack([x, np.ones(len(x))]).T
    y = np.array([res['psi'].entanglement_entropy()[0] for res in results.values()])
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print('TEE: ' + str(c))
    # plt.xlim([0,x[-1]])
    # plt.ylim([-1,y[-1]])
    # plt.plot(x, y, 'o', label='Original data', markersize=10)
    # plt.plot(x, m*x + c, 'r', label='Fitted line')
    # plt.legend()
    # plt.show()
    return c

# results_Ly = {}

# for Ly in range(5, 8):
#     results_Ly[Ly] = run_DMRG(Ly, 5., 5.,h = 0.38,g = 0.38)

# Stopo(results_Ly)

Stopos = {}
vevs = {}
paramSweep = [0.37] #transition at 0.34
for x in paramSweep:
    results_Ly = {}
    for Ly in range(8,9):
        results_Ly[Ly] = run_DMRG(Ly, 5., 4.5,h = x,g = x)
        psi = results_Ly[Ly]['psi']
        print("parameter: " + str(x) + " size " + str(Ly))
        print("sigmax - sigmaz")
        print( str(psi.expectation_value("Sigmax")))
        print( str(psi.expectation_value("Sigmaz")))
    Stopos[x] = Stopo(results_Ly)

# psi_list = [res['psi'] for res in results_loops.values()]
# overlaps= [[psi_i.overlap(psi_j) for psi_j in psi_list] for psi_i in psi_list]
# print("overlaps")
# print(np.array(overlaps))


# data = [psi._B for psi in psi_list]
# np.save('ohbaby.npy',data)




# model_params['order'] = 'Cstyle' # The effect doesn't appear with the "default" ordering for the toric code.
# # This is also a hint that you need bond 0: you want something independent of what order you choose
# # inside the MPS unit cell.

# print( psi.expectation_value("Sigmax") )
