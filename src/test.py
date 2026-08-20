import copy
import os
import pickle

from astropy import units as u

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, ChandrasekharDynamicalFrictionForce, HernquistPotential, MovingObjectPotential

from hvs import initialize_hvs_events, integrate_lmc_hvs_orbits_vectorized

import numpy as np

if __name__ == "__main__":
    lookback_time = 2e8 * u.yr
    events = initialize_hvs_events(lookback_time=lookback_time, hvs_rate=1e-4 * u.yr**-1, alpha=2.35, m_min=2 * u.Msun, m_max=100 * u.Msun)

    # make directory if it doesn't exist
    dir = "data/lmc_test/alpha2.35"
    os.makedirs(dir, exist_ok=True)
    # save events to a file
    events.write(f'{dir}/events.fits', overwrite=True)

    hvs_events = events[events['ejected']]
    hvs_events.write(f'{dir}/hvs_events.fits', overwrite=True)

    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=1e11 * u.Msun, rhm=5.0 * u.kpc, dens=MWPotential2014
    )
    o_lmc = Orbit.from_name("LMC")

    # increase halo mass by 50% so that LMC is bound
    mwp = copy.deepcopy(MWPotential2014)
    mwp[2] *= 1.5

    n_steps = 1000
    t_grid = np.linspace(
        0, -lookback_time.to_value(u.yr),
        n_steps,
    ) * u.yr

    # integrate the LMC orbit including dynamical friction.
    o_lmc.integrate(t_grid, mwp + cdf, progressbar=True, method="rk4_c")
    print(o_lmc.R())


    # orbits, lmc_orbit = integrate_lmc_hvs_orbits_vectorized(hvs_events, mwp, o_lmc, cdf, lookback_time=lookback_time, n_steps=100)

    # with open(f'{dir}/hvs_orbits.pkl', 'wb') as f:
    #     pickle.dump(orbits, f)
    # with open(f'{dir}/lmc_orbit.pkl', 'wb') as f:
    #     pickle.dump(lmc_orbit, f)