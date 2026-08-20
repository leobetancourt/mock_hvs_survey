from tqdm import tqdm

import numpy as np

from astropy.table import QTable
import astropy.units as u

from galpy.orbit import Orbit

from helpers import random_dir_on_sphere, spherical_to_cylindrical, angle_between, sample_power_law, roche_lobe_radius_fraction, main_sequence_radius

def IMF(mass_min, mass_max, N, alpha=2.35):
    """
    Sample stellar masses from an IMF between mass_min and mass_max. Default is Salpeter IMF with alpha=2.35.
    """
    # Inverse transform sampling
    r = np.random.uniform(0, 1, N)
    mass = ((mass_max**(1 - alpha) - mass_min**(1 - alpha)) * r + mass_min**(1 - alpha))**(1 / (1 - alpha))
    return mass

def D(a_bin, r_peri, m_bin, M_bh):
    """
    Normalized distance of closest approach
    """
    return (r_peri / a_bin) * (1e6 * m_bin / (2 * M_bh))**(1 / 3)

def f_R(D):
    # Polynomial fit to ejection velocity distribution from Bromley et al. (2006)
    return 0.774 + (0.0204 + (-6.23e-4 + (7.62e-6 + (-4.24e-8 + 8.62e-11 * D) * D) * D) * D) * D

def P_ej(D):
    return np.where(D < 175, 1 - (D / 175), 0)

def v_ej_hvs(a_bin, r_peri, m_bin, M_bh=4e6 * u.Msun):
    D_ = D(a_bin, r_peri, m_bin, M_bh)
    f_R_ = f_R(D_)
    P_ej_ = P_ej(D_)
    v_hvs = 1370 * u.km / u.s * (a_bin / (0.1 * u.au)) ** (-1/2) * (m_bin / (1 * u.Msun))**(1/3) * (M_bh / (4e6 * u.Msun))**(1/6) * f_R_

    # eject stars according to ejection probability
    u_rand = np.random.rand(len(a_bin))
    v_ej = np.zeros(len(a_bin)) * u.km / u.s
    mask = u_rand < P_ej_
    v_ej[mask] = v_hvs[mask]
    return v_hvs, mask

def initialize_hvs_events(t_start=-2e8*u.yr, t_end=0*u.yr, hvs_rate=1e-4 * u.yr**-1, alpha=2.35, 
                          m_min=0.1 * u.Msun, m_max=100 * u.Msun, M_bh=4e6 * u.Msun, 
                          a_bin_min=None, a_bin_max=None, a_bin_alpha=-1, 
                          r_peri_min=None, r_peri_max=None, r_peri_alpha=2, v_ej_max=None):
    """
    Initialize hypervelocity star (HVS) ejection events over a specified lookback time, given a binary injection rate.

    Returns:
    - hvs_events: Astropy Table containing the following columns:
        - t_ejection: Ejection times of HVS events (negative, in years before observation)
        - m_star_prim: Masses of primary stars in the binaries (in Msun)
        - m_star_sec: Masses of secondary stars in the binaries (in Msun)
        - a_bin: Binary separations (in au)
        - r_peri: Pericenter distances for the ejections (in au)
        - v_ejection: Ejection velocities of HVS events (in km/s)
    """
    from astropy.constants import G, c
    binary_injection_rate = hvs_rate
    integration_time = t_end - t_start

    # generate HVS ejection times from a Poisson process
    N_ejection = np.random.poisson(binary_injection_rate * integration_time)
    t_ejection_arr = np.sort(np.random.uniform(t_start.to(u.yr).value, t_end.to(u.yr).value, N_ejection)) * u.yr

    # implement delay time between binary formation and ejection
    t_delay_max = 150 * u.Myr
    t_delay_arr = np.random.uniform(0, t_delay_max.to(u.yr).value, N_ejection) * u.yr

    m_star_min = m_min
    m_star_max = m_max
    # sample primary star from IMF
    m_star_prim_arr = IMF(m_star_min, m_star_max, N_ejection, alpha)
    # sample secondary star from uniform distribution in mass ratio between 0 and 1
    q = np.random.rand(N_ejection)
    m_star_sec_arr = q * m_star_prim_arr
    m_bin_arr = m_star_prim_arr + m_star_sec_arr

    u_rand = np.random.rand(N_ejection)
    # randomly choose which star is ejected (primary or secondary) with equal probability
    m_ejected = np.where(u_rand < 0.5, m_star_prim_arr, m_star_sec_arr)

    u_rand = np.random.rand(N_ejection)
    # assume log-uniform distribution of binary separations (Opik's law)
    q1 = m_star_prim_arr / m_star_sec_arr
    q2 = m_star_sec_arr / m_star_prim_arr
    R1 = main_sequence_radius(m_star_prim_arr)
    R2 = main_sequence_radius(m_star_sec_arr)
    f1 = roche_lobe_radius_fraction(q1)
    f2 = roche_lobe_radius_fraction(q2)
    a_bin_min = np.maximum(R1 / f1, R2 / f2)
    a_bin_max = np.ones(N_ejection) * 4 * u.au
    a_bin_arr = sample_power_law(a_bin_min, a_bin_max, alpha=a_bin_alpha, size=N_ejection)

    if r_peri_min is None:
        r_peri_min = R2 * (M_bh / m_star_sec_arr)**(1/3) # tidal disruption radius for the secondary
    if r_peri_max is None:
        r_peri_max = a_bin_arr * (3 * M_bh / (m_star_prim_arr + m_star_sec_arr))**(1/3) # tidal disruption radius for the binary
    R_Sch = 2 * G * M_bh / c**2
    # ensure r_peri_min is not less than the Schwarzschild radius
    r_peri_min = np.where(r_peri_min < R_Sch, R_Sch, r_peri_min)
    r_peri_max = np.where(r_peri_max < R_Sch, R_Sch, r_peri_max)

    r_peri_arr = sample_power_law(r_peri_min, r_peri_max, alpha=r_peri_alpha, size=N_ejection)
    # for the tightest binaries, might have r_peri_min > r_peri_max, so mask those
    bad_mask = r_peri_min >= r_peri_max
    
    # r_peri_arr = r_peri_max # put all at binary disruption radius
    # r_peri_arr = r_peri_min # put all at secondary disruption radius

    v_hvs_arr, mask = v_ej_hvs(a_bin_arr, r_peri_arr, m_bin_arr, M_bh)
    mask = mask & ~bad_mask
    if v_ej_max is not None:
        v_hvs_arr = np.where(v_hvs_arr > v_ej_max, v_ej_max, v_hvs_arr)

    # assemble all events into an Astropy Table
    hvs_events = QTable([t_ejection_arr.to(u.yr), t_delay_arr.to(u.yr), m_star_prim_arr.to(u.Msun), m_star_sec_arr.to(u.Msun), m_ejected.to(u.Msun), a_bin_arr.to(u.au), r_peri_arr.to(u.au), v_hvs_arr.to(u.km/u.s), mask],
                       names=['t_ejection', 't_delay', 'm_star_prim', 'm_star_sec', 'm_ejected', 'a_bin', 'r_peri', 'v_ejection', 'ejected'])
    return hvs_events

def integrate_hvs_orbits_vectorized(hvs_events, potential, n_steps=1000, pbar=True):
    """
    Integrate HVS orbits in a vectorized galpy Orbit.

    Returns
    -------
    o : galpy.orbit.Orbit
        Multi-object Orbit containing all HVSs.
    flight_times : astropy Quantity
        Flight time of each star from ejection to present.
    """
    vxvv = []

    for event in tqdm(hvs_events, desc="Setting up initial conditions"):
        r_peri = event["r_peri"]
        v_ejection = event["v_ejection"]

        r0 = r_peri
        theta0, phi0 = random_dir_on_sphere()

        vr0 = v_ejection
        vtheta0 = 0 * u.km / u.s
        vphi0 = 0 * u.km / u.s

        r0, phi0, z0, vr0, vz0, vphi0 = spherical_to_cylindrical(
            r0, theta0, phi0, vr0, vtheta0, vphi0
        )

        vxvv.append([r0.to(u.kpc), vr0.to(u.km/u.s), vphi0.to(u.km/u.s), z0.to(u.kpc), vz0.to(u.km/u.s), phi0.to(u.rad)])

    orbits = Orbit(vxvv, ro=8.12, vo=220)
    # set physical units for the orbits, so it always returns orbital parameters with units
    orbits.turn_physical_on()

    t_start = hvs_events["t_ejection"].to_value(u.yr)
    t_stop = np.zeros_like(t_start)
    t_grid = np.linspace(t_start, t_stop, n_steps, axis=-1) * u.yr
    orbits.integrate(t_grid, potential, method="rk4_c", progressbar=pbar)
    return orbits

def integrate_lmc_hvs_orbits_vectorized(
    hvs_events,
    mw_potential,
    lmc_orbit,
    df_potential,
    t_start=-2e8 * u.yr,
    t_end=0 * u.yr,
    n_steps=1000,
    pbar=True,
):
    """
    Integrate HVSs ejected from the LMC.

    Stars are launched from the LMC center at their ejection times, with
    velocities defined in the LMC rest frame.
    """
    from galpy.potential import HernquistPotential, MovingObjectPotential, MiyamotoNagaiPotential

    if t_start == 0 * u.yr:
        t_grid = np.linspace(
            0, t_end.to_value(u.yr),
            n_steps,
        ) * u.yr
        lmc_orbit.integrate(t_grid, mw_potential + df_potential, progressbar=pbar, method="rk4_c")
    elif t_end == 0 * u.yr:
        t_grid = np.linspace(
            0, t_start.to_value(u.yr),
            n_steps,
        ) * u.yr
        lmc_orbit.integrate(t_grid, mw_potential + df_potential, progressbar=pbar, method="rk4_c")
    else: # need to integrate forwards and backwards separately
        t_grid = np.linspace(
            0, t_end.to_value(u.yr),
            n_steps,
        ) * u.yr
        lmc_orbit.integrate(t_grid, mw_potential + df_potential, progressbar=pbar, method="rk4_c")
        t_grid = np.linspace(
            0, t_start.to_value(u.yr),
            n_steps,
        ) * u.yr
        lmc_orbit.integrate(t_grid, mw_potential + df_potential, progressbar=pbar, method="rk4_c")

    # set up LMC potential
    lmc_disk = MiyamotoNagaiPotential(amp=2.5e9 * u.Msun, a=1.5 * u.kpc, b=0.5 * u.kpc)
    lmc_pot = HernquistPotential(amp=2e11*u.Msun, a=15.0*u.kpc)
    lmc_moving_pot = MovingObjectPotential(lmc_orbit, pot=[lmc_disk, lmc_pot])

    vxvv = []

    t_ejection = hvs_events['t_ejection']
    theta_ej = []
    for event, t_eject in tqdm(
        zip(hvs_events, t_ejection),
        total=len(hvs_events),
        desc="Setting up LMC-ejected initial conditions",
    ):
        r_peri = event["r_peri"].to(u.kpc)
        v_ejection = event["v_ejection"].to(u.km / u.s)

        # LMC phase-space coordinates at ejection time
        R_lmc = lmc_orbit.R(t_eject, quantity=True).to(u.kpc)
        z_lmc = lmc_orbit.z(t_eject, quantity=True).to(u.kpc)
        phi_lmc = lmc_orbit.phi(t_eject, quantity=True).to(u.rad)

        vR_lmc = lmc_orbit.vR(t_eject, quantity=True).to(u.km / u.s)
        vz_lmc = lmc_orbit.vz(t_eject, quantity=True).to(u.km / u.s)
        vT_lmc = lmc_orbit.vT(t_eject, quantity=True).to(u.km / u.s)

        # Random launch direction in Galactocentric Cartesian axes.
        # theta is polar angle from +z, phi is azimuth in x-y.
        theta, phi = random_dir_on_sphere()

        dx = r_peri * np.sin(theta) * np.cos(phi)
        dy = r_peri * np.sin(theta) * np.sin(phi)
        dz = r_peri * np.cos(theta)

        dvx = v_ejection * np.sin(theta) * np.cos(phi)
        dvy = v_ejection * np.sin(theta) * np.sin(phi)
        dvz = v_ejection * np.cos(theta)

        cos_lmc = np.cos(phi_lmc)
        sin_lmc = np.sin(phi_lmc)

        # LMC cylindrical -> Cartesian
        x_lmc = R_lmc * cos_lmc
        y_lmc = R_lmc * sin_lmc
        z_lmc = z_lmc

        vx_lmc = vR_lmc * cos_lmc - vT_lmc * sin_lmc
        vy_lmc = vR_lmc * sin_lmc + vT_lmc * cos_lmc
        vz_lmc = vz_lmc

        # calculate angle between ejected star and LMC's velocity vector
        v_star = np.array([dvx.to_value(u.km/u.s), dvy.to_value(u.km/u.s), dvz.to_value(u.km/u.s)])
        v_lmc = np.array([vx_lmc.to_value(u.km/u.s), vy_lmc.to_value(u.km/u.s), vz_lmc.to_value(u.km/u.s)])
        angle_rad = angle_between(v_star, v_lmc)
        theta_ej.append(angle_rad)

        # Star Cartesian phase-space coordinates
        x = x_lmc + dx
        y = y_lmc + dy
        z = z_lmc + dz

        vx = vx_lmc + dvx
        vy = vy_lmc + dvy
        vz = vz_lmc + dvz

        # Cartesian -> cylindrical at the star's actual position
        R = np.sqrt(x**2 + y**2).to(u.kpc)
        phi_star = np.arctan2(y, x).to(u.rad)

        cos_star = np.cos(phi_star)
        sin_star = np.sin(phi_star)

        vR = (vx * cos_star + vy * sin_star).to(u.km / u.s)
        vT = (-vx * sin_star + vy * cos_star).to(u.km / u.s)

        vxvv.append([
            R,
            vR,
            vT,
            z.to(u.kpc),
            vz.to(u.km / u.s),
            phi_star,
        ])

    orbits = Orbit(vxvv, ro=8.12, vo=220)
    orbits.turn_physical_on()

    t_start_indiv = hvs_events["t_ejection"].to_value(u.yr)
    t_grid_indiv = np.linspace(t_start_indiv, t_end.to_value(u.yr), n_steps, axis=-1) * u.yr
    orbits.integrate(t_grid_indiv, mw_potential + lmc_moving_pot, progressbar=pbar, method="rk4_c")
    return orbits, lmc_orbit, theta_ej