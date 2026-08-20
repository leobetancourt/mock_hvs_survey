import copy
import os
import pickle

from astropy import units as u

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, ChandrasekharDynamicalFrictionForce

from hvs import initialize_hvs_events, integrate_lmc_hvs_orbits_vectorized


if __name__ == "__main__":
    # number of simulations to run
    N = 1
    for i in range(N):
        print(f"Running simulation {i+1}/{N}...")
        t_start = -200e6 * u.yr
        t_end = 0e6 * u.yr
        # a_bin_min = 0.03 * u.au
        # a_bin_max = 4 * u.au
        rate = 1e-4 * u.yr**-1
        events = initialize_hvs_events(t_start=t_start, t_end=t_end, hvs_rate=rate, alpha=1.6, 
                                    m_min=0.1 * u.Msun, m_max=100 * u.Msun, M_bh=6e5 * u.Msun)

        hvs_events = events[events['ejected']]

        cdf = ChandrasekharDynamicalFrictionForce(
            GMs=1e11 * u.Msun, rhm=5.0 * u.kpc, dens=MWPotential2014
        )
        o_lmc = Orbit.from_name("LMC", ro=8.12*u.kpc, vo=220*u.km/u.s)

        # increase halo mass by 50% so that LMC is bound
        mwp = copy.deepcopy(MWPotential2014)
        mwp[2] *= 1.5
        orbits, lmc_orbit, theta_ej = integrate_lmc_hvs_orbits_vectorized(hvs_events, mwp, o_lmc, cdf, t_start=t_start, t_end=t_end, n_steps=100)

        hvs_events['theta_ej'] = theta_ej

        # save simulation data
        # make directory if it doesn't exist
        dir = f"simulations/lmc_test/"
        os.makedirs(dir, exist_ok=True)
        # save events to a file
        events.write(f'{dir}/events.fits', overwrite=True)
        hvs_events.write(f'{dir}/hvs_events.fits', overwrite=True)

        with open(f'{dir}/hvs_orbits.pkl', 'wb') as f:
            pickle.dump(orbits, f)
        with open(f'{dir}/lmc_orbit.pkl', 'wb') as f:
            pickle.dump(lmc_orbit, f)