import os
import pickle

from astropy import units as u

from galpy.potential import MWPotential2014

from hvs import initialize_hvs_events, integrate_hvs_orbits_vectorized


if __name__ == "__main__":
    N = 1
    for i in range(N):
        print(f"Running simulation {i+1}/{N}...")
        t_start = -200e6 * u.yr
        t_end = 0e6 * u.yr
        a_bin_min = 0.03 * u.au
        a_bin_max = 3 * u.au
        rate = 1e-4 * u.yr**-1
        events = initialize_hvs_events(t_start=t_start, t_end=t_end, hvs_rate=rate, alpha=1.6, 
                                    m_min=0.1 * u.Msun, m_max=100 * u.Msun, M_bh=4e6 * u.Msun,
                                    a_bin_min=a_bin_min, a_bin_max=a_bin_max)
        dir = f"simulations/mw_test/r_peri_distr/"
        os.makedirs(dir, exist_ok=True)
        # save events to a file
        events.write(f'{dir}/events.fits', overwrite=True)

        hvs_events = events[events['ejected']]
        hvs_events.write(f'{dir}/hvs_events.fits', overwrite=True)

        orbits = integrate_hvs_orbits_vectorized(hvs_events, MWPotential2014, n_steps=100, pbar=True)

        # make directory if it doesn't exist
        with open(f'{dir}/hvs_orbits.pkl', 'wb') as f:
            pickle.dump(orbits, f)
