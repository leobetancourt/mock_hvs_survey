from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import pickle

from astropy import units as u
from astropy.io import fits
from astropy.table import QTable

import minimint

from galpy.orbit import Orbit

import healpy as hp

from helpers import absolute_to_apparent_magnitude, v_esc

def load_hvs_simulation_data(dir):
    events = QTable.read(f'{dir}/events.fits', format='fits')
    hvs_events = QTable.read(f'{dir}/hvs_events.fits', format='fits')

    with open(f'{dir}/hvs_orbits.pkl', 'rb') as f:
        orbits = pickle.load(f)

    return events, hvs_events, orbits

def get_hvs_at_time(events, orbits, t):
    R = []
    vR = []
    vT = []
    z = []
    vz = []
    phi = []
    events_mask = []

    for i in range(len(orbits)):
        oi = orbits[i]

        ti = oi.time(quantity=True)
        t_i = t.to(ti.unit)

        if t_i < np.nanmin(ti) or t_i > np.nanmax(ti):
            events_mask.append(False)
            continue

        R.append(oi.R(t_i, quantity=True).to(u.kpc))
        vR.append(oi.vR(t_i, quantity=True).to(u.km / u.s))
        vT.append(oi.vT(t_i, quantity=True).to(u.km / u.s))
        z.append(oi.z(t_i, quantity=True).to(u.kpc))
        vz.append(oi.vz(t_i, quantity=True).to(u.km / u.s))
        phi.append(oi.phi(t_i, quantity=True).to(u.rad))
        events_mask.append(True)

    if len(R) == 0:
        return events[events_mask], Orbit(vxvv=None, ro=orbits._ro, vo=orbits._vo)

    vxvv = [
        u.Quantity(R),
        u.Quantity(vR),
        u.Quantity(vT),
        u.Quantity(z),
        u.Quantity(vz),
        u.Quantity(phi),
    ]

    orbits_t = Orbit(vxvv=vxvv, ro=orbits._ro, vo=orbits._vo)
    orbits_t.turn_physical_on()
    return events[events_mask], orbits_t

def get_iso_data(hvs_events, sky_coords, t):
    # query MIST isochrones for stellar parameters and Gaia photometry, at present-day HVS distances.
    m_star = hvs_events['m_ejected'].to_value(u.Msun)
    # calculate age of star based on ejection time and delay time
    age = (t - hvs_events['t_ejection']) + hvs_events['t_delay']
    log_age = np.log10(age.to_value(u.yr)) # should probably implement a t_delay later
    feh = -1.0
    dist_hvs_sun = sky_coords.distance

    # set up minimint interpolator for Gaia filters
    filters = ["Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3', "SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i"]
    ii = minimint.Interpolator(filters)
    isos = ii(m_star, log_age, feh)
    phase = isos['phase']
    SDSS_u = absolute_to_apparent_magnitude(isos['SDSS_u'], dist_hvs_sun.to_value(u.pc))
    SDSS_g = absolute_to_apparent_magnitude(isos['SDSS_g'], dist_hvs_sun.to_value(u.pc))
    SDSS_r = absolute_to_apparent_magnitude(isos['SDSS_r'], dist_hvs_sun.to_value(u.pc))
    SDSS_i = absolute_to_apparent_magnitude(isos['SDSS_i'], dist_hvs_sun.to_value(u.pc))

    G = absolute_to_apparent_magnitude(isos['Gaia_G_EDR3'], dist_hvs_sun.to_value(u.pc))
    BP_RP = isos['Gaia_BP_EDR3'] - isos['Gaia_RP_EDR3']

    return SDSS_u, SDSS_g, SDSS_r, SDSS_i, G, BP_RP, phase

def plot_initial_conditions(events, title=None, filename=None):
    a_bin = events['a_bin'].to_value(u.au)
    r_peri = events['r_peri'].to_value(u.au)
    v_ejection = events['v_ejection'].to_value(u.km / u.s)
    ejected = events['ejected']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # a_bin: log-spaced bins for Opik's law
    a_bins = np.logspace(np.log10(a_bin.min()), np.log10(a_bin.max()), 40)
    counts, edges = np.histogram(a_bin, bins=a_bins)
    axes[0, 0].stairs(counts, edges, linewidth=1, color='k', alpha=0.5, label='all')

    counts, edges = np.histogram(a_bin[ejected], bins=a_bins)
    axes[0, 0].stairs(counts, edges, linewidth=1, color='k', label='launched')

    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"$a_\mathrm{bin}\rm\ [au]$")
    axes[0, 0].set_ylabel(r"$N$")
    axes[0, 0].set_yscale("log")
    # axes[0].set_title(r"Binary separation")

    # r_peri
    r_bins = np.linspace(r_peri.min(), r_peri.max(), 40)
    counts, edges = np.histogram(r_peri, bins=r_bins)
    axes[1, 0].stairs(counts, edges, linewidth=1, color='k', alpha=0.5, label='all')

    counts, edges = np.histogram(r_peri[ejected], bins=r_bins)
    axes[1, 0].stairs(counts, edges, linewidth=1, color='k', label='launched')
    axes[1, 0].set_xlabel(r"$r_\mathrm{peri}\rm\ [au]$")
    axes[1, 0].set_ylabel(r"$N$")
    axes[1, 0].set_yscale("log")
    # axes[1].set_title(r"Pericenter distance")

    # v_ejection
    v_min = v_ejection.min()
    v_max = 10000
    v_bins = np.linspace(v_min, v_max, 40)
    counts, edges = np.histogram(v_ejection, bins=v_bins)
    axes[0, 1].stairs(counts, edges, linewidth=1, color='k', alpha=0.5, label='all binaries')

    counts, edges = np.histogram(v_ejection[ejected], bins=v_bins)
    axes[0, 1].stairs(counts, edges, linewidth=1, color='k', label='HVS progenitors')

    axes[0, 1].set_xlabel(r"$v_\mathrm{ejection}\rm\ [km\ s^{-1}]$")
    axes[0, 1].set_ylabel(r"$N$")
    axes[0, 1].set_xlim(v_min, v_max)
    # axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    # axes[2].set_title(r"Ejection velocity")

    m_star = events['m_ejected'].to_value(u.Msun)
    m_bins = np.linspace(m_star.min(), m_star.max(), 40)
    counts, edges = np.histogram(m_star, bins=m_bins)
    axes[1, 1].stairs(counts, edges, linewidth=1, color='k', alpha=0.5, label='all binaries')

    counts, edges = np.histogram(m_star[ejected], bins=m_bins)
    axes[1, 1].stairs(counts, edges, linewidth=1, color='k', alpha=1, label='Hills binaries')
    axes[1, 1].set_xlabel(r"$M_\mathrm{*}\rm\ [M_\odot]$")
    axes[1, 1].set_ylabel(r"$N$")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    # axes[3].set_title(r"Stellar mass")

    if title: fig.suptitle(title)
    if filename: fig.savefig(filename)
    plt.show()

from functools import lru_cache
from astropy.coordinates import SkyCoord

@lru_cache(maxsize=8)
def load_footprint_mollweide_grid(
    path,
    plot_frame="icrs",
    map_frame="icrs",
    n_lon=1440,
    n_lat=720,
    threshold=0,
):
    """
    Return lon, lat, mask for plotting a HEALPix footprint under a Mollweide plot.

    plot_frame:
        Coordinate frame shown in the plot: "icrs" or "galactic".

    map_frame:
        Coordinate frame of the HEALPix map: "icrs" or "galactic".

    Returns:
        lon, lat in radians, and a mask array where footprint pixels are 1
        and empty pixels are NaN.
    """
    hpx = hp.read_map(path, verbose=False)
    nside = hp.get_nside(hpx)

    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Your plots use x = -longitude_wrapped, so convert plot x back to sky longitude.
    sky_lon_wrapped = -np.degrees(lon2d)
    sky_lon = (sky_lon_wrapped + 360) % 360
    sky_lat = np.degrees(lat2d)

    plot_frame = plot_frame.lower()
    map_frame = map_frame.lower()

    if plot_frame == map_frame:
        map_lon = sky_lon
        map_lat = sky_lat
    else:
        if plot_frame == "icrs":
            coord = SkyCoord(ra=sky_lon * u.deg, dec=sky_lat * u.deg, frame="icrs")
        elif plot_frame == "galactic":
            coord = SkyCoord(l=sky_lon * u.deg, b=sky_lat * u.deg, frame="galactic")
        else:
            raise ValueError("plot_frame must be 'icrs' or 'galactic'")

        if map_frame == "icrs":
            map_coord = coord.icrs
            map_lon = map_coord.ra.deg
            map_lat = map_coord.dec.deg
        elif map_frame == "galactic":
            map_coord = coord.galactic
            map_lon = map_coord.l.deg
            map_lat = map_coord.b.deg
        else:
            raise ValueError("map_frame must be 'icrs' or 'galactic'")

    theta = np.radians(90 - map_lat)
    phi = np.radians(map_lon)

    pix = hp.ang2pix(nside, theta, phi)
    footprint_values = hpx[pix]

    footprint_mask = np.where(footprint_values > threshold, 1.0, np.nan)

    return lon, lat, footprint_mask

def get_footprint_mask(footprint_file, ra_hvs, dec_hvs):
    """
    Get the footprint mask for the given RA and Dec coordinates.
    """

    with fits.open(footprint_file, memmap=True) as hdul:
        nside = hdul[1].header["NSIDE"]
        nest = hdul[1].header["ORDERING"].strip().upper() == "NESTED"
        data = hdul[1].data.field(0)

        theta = np.radians(90.0 - dec_hvs.to_value(u.deg))
        phi = np.radians(ra_hvs.to_value(u.deg))

        isnan = np.isnan(theta) | np.isnan(phi)

        theta_safe = np.where(isnan, 0.0, theta)
        phi_safe = np.where(isnan, 0.0, phi)

        pix = hp.ang2pix(nside, theta_safe, phi_safe, nest=nest)

        rows = pix // 1024
        cols = pix % 1024

        mask = data[rows, cols] > 0

        return np.where(isnan, False, mask)

def get_hvs_survey_cuts(sky_coords, phase, SDSS_u, SDSS_g, SDSS_r, SDSS_i):
    import astropy.coordinates as coord
    from helpers import v_esc

    if len(SDSS_u) == 0:
        cuts = defaultdict(list)
        cuts['velocity_cut'] = np.zeros(1, dtype=bool)
        cuts['ms_cut'] = np.zeros(1, dtype=bool)
        cuts['mag_color_cut'] = np.zeros(1, dtype=bool)
        cuts['footprint_cut'] = np.zeros(1, dtype=bool)
        cuts['observable'] = np.zeros(1, dtype=bool)
        return np.array([]), cuts

    sky_coords_gal = sky_coords.transform_to(coord.Galactocentric())
    r = coord.CartesianRepresentation(sky_coords_gal.x, sky_coords_gal.y, sky_coords_gal.z)
    v = coord.CartesianRepresentation(sky_coords_gal.v_x, sky_coords_gal.v_y, sky_coords_gal.v_z)

    v_rad_gal = (r.dot(v) / r.norm()).to(u.km / u.s)
    v_tot_gal = np.sqrt(sky_coords_gal.v_x**2 + sky_coords_gal.v_y**2 + sky_coords_gal.v_z**2).to(u.km / u.s)
    v_esc_gal = v_esc(r.norm().to_value(u.kpc))

    ra_hvs = sky_coords.ra
    dec_hvs = sky_coords.dec

    velocity_cut = v_tot_gal > v_esc_gal # must be unbound
    ms_cut = phase == 0 # must be on main sequence
    g_r = SDSS_g - SDSS_r
    u_g = SDSS_u - SDSS_g
    r_i = SDSS_r - SDSS_i
    mag_color_cut = (SDSS_g > 17) & (SDSS_g < 20.25) & \
                    ((g_r > -0.4) & (g_r < (-0.43*u_g + 0.18))) & \
                    ((r_i > -0.5) & (r_i < 0)) & \
                    ((u_g > 2.2*g_r + 0.1) & (u_g < 1.07))
    footprint_cut = get_footprint_mask("sdss_dr8_footprint.fits", ra_hvs, dec_hvs)

    observable = velocity_cut & ms_cut & mag_color_cut & footprint_cut

    cuts = defaultdict(list)
    cuts['velocity_cut'] = velocity_cut
    cuts['ms_cut'] = ms_cut
    cuts['mag_color_cut'] = mag_color_cut
    cuts['footprint_cut'] = footprint_cut
    cuts['observable'] = observable
    return observable, cuts

def get_euclid_dr1_survey_cuts(sky_coords, phase, Gaia_G, Gaia_BP_RP):
    import astropy.coordinates as coord
    from helpers import v_esc

    sky_coords_gal = sky_coords.transform_to(coord.Galactocentric())
    r = coord.CartesianRepresentation(sky_coords_gal.x, sky_coords_gal.y, sky_coords_gal.z)
    v = coord.CartesianRepresentation(sky_coords_gal.v_x, sky_coords_gal.v_y, sky_coords_gal.v_z)

    v_rad_gal = (r.dot(v) / r.norm()).to(u.km / u.s)
    v_tot_gal = np.sqrt(sky_coords_gal.v_x**2 + sky_coords_gal.v_y**2 + sky_coords_gal.v_z**2).to(u.km / u.s)
    v_esc_gal = v_esc(r.norm().to_value(u.kpc))

    # get heliocentric RV
    v_helio = sky_coords.radial_velocity.to(u.km / u.s)

    ra_hvs = sky_coords.ra
    dec_hvs = sky_coords.dec

    velocity_cut = v_helio > 600 * u.km / u.s
    unbound_cut = v_tot_gal > v_esc_gal
    ms_cut = phase == 0 # must be on main sequence
    mag_cut = Gaia_G < 19
    footprint_cut = get_footprint_mask("euclid_dr1_footprint.fits", ra_hvs, dec_hvs)

    observable = velocity_cut & mag_cut & footprint_cut

    cuts = defaultdict(list)
    cuts['velocity_cut'] = velocity_cut
    cuts['unbound_cut'] = unbound_cut
    cuts['ms_cut'] = ms_cut
    cuts['mag_cut'] = mag_cut
    cuts['footprint_cut'] = footprint_cut
    cuts['observable'] = observable
    return observable, cuts

def plot_galactic_distribution(sky_coords, mask=None, footprint_file="sdss_dr8_footprint.fits", fig=None, ax=None, title=None, show=False):
    from matplotlib.colors import ListedColormap

    l_hvs, b_hvs = sky_coords.galactic.l, sky_coords.galactic.b

    # wrap longitude from [0, 360] to [-180, 180]
    l_wrapped = ((l_hvs.to_value(u.deg) + 180) % 360) - 180

    l_rad = np.radians(-l_wrapped)
    b_rad = np.radians(b_hvs.to_value(u.deg))

    if fig is None or ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection="mollweide")

    lon, lat, footprint = load_footprint_mollweide_grid(footprint_file, plot_frame="galactic", map_frame="icrs")
    gray_cmap = ListedColormap(["0.85"])
    ax.pcolormesh(
        lon,
        lat,
        np.where(footprint > 0, 1.0, np.nan),
        shading="nearest",
        cmap=gray_cmap,
        alpha=1.0,
        rasterized=True,
        zorder=0,
        label='survey footprint'
    )

    ax.scatter(
        l_rad,
        b_rad,
        s=0.5,
        alpha=0.5,
        facecolors="none",
        edgecolors="black",
        linewidths=0.1,
        rasterized=True,
    )

    # if mask provided, plot these in blue
    if mask is not None and isinstance(l_rad, np.ndarray) and len(mask) > 0:
        ax.scatter(
            l_rad[mask],
            b_rad[mask],
            s=3,
            alpha=1,
            linewidths=0.5,
            edgecolors="blue",
            facecolors="none",
            rasterized=True,
            label=r'$\mathrm{observable}$'
        )

    ax.set_xlabel(r"$\mathrm{Galactic\ longitude}\ l$")
    ax.set_ylabel(r"$\mathrm{Galactic\ latitude}\ b$")
    ax.grid(True, alpha=0.3)

    tick_labels = [
        r"$150^\circ$", r"$120^\circ$", r"$90^\circ$",
        r"$60^\circ$", r"$30^\circ$", r"$0^\circ$",
        r"$330^\circ$", r"$300^\circ$", r"$270^\circ$",
        r"$240^\circ$", r"$210^\circ$",
    ]
    ax.set_xticklabels(tick_labels, fontsize=8)
    tick_labels = [r"$-75^\circ$", r"$-60^\circ$", r"$-45^\circ$", r"$-30^\circ$", r"$-15^\circ$", r"$0^\circ$", r"$15^\circ$", r"$30^\circ$", r"$45^\circ$", r"$60^\circ$", r"$75^\circ$"]
    ax.set_yticklabels(tick_labels, fontsize=8)
    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if show: plt.show()
    return fig, ax

def plot_galactic_velocity(sky_coords, cut, title=None, footprint_file="sdss_dr8_footprint.fits", show=False):
    from matplotlib.colors import ListedColormap

    l_hvs, b_hvs = sky_coords.galactic.l, sky_coords.galactic.b

    # wrap longitude from [0, 360] to [-180, 180]
    l_wrapped = ((l_hvs.to_value(u.deg) + 180) % 360) - 180

    l_rad = np.radians(-l_wrapped)
    b_rad = np.radians(b_hvs.to_value(u.deg))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="mollweide")

    lon, lat, footprint = load_footprint_mollweide_grid(footprint_file, plot_frame="galactic", map_frame="icrs")
    gray_cmap = ListedColormap(["0.85"])
    ax.pcolormesh(
        lon,
        lat,
        np.where(footprint > 0, 1.0, np.nan),
        shading="nearest",
        cmap=gray_cmap,
        alpha=1.0,
        rasterized=True,
        zorder=0,
    )

    # color and scale by radial velocity in Galactocentric frame
    sc_gal = sky_coords.transform_to('galactocentric')
    r_gal = np.sqrt(sc_gal.x**2 + sc_gal.y**2 + sc_gal.z**2)
    v_tot_gal = np.sqrt(sc_gal.v_x**2 + sc_gal.v_y**2 + sc_gal.v_z**2).to(u.km / u.s)
    # radial velocity in Galactocentric frame
    v_rad_gal = (sc_gal.x * sc_gal.v_x + sc_gal.y * sc_gal.v_y + sc_gal.z * sc_gal.v_z) / r_gal
    v_esc_gal = v_esc(r_gal.to_value(u.kpc))
    diff = (v_rad_gal - v_esc_gal).to_value(u.km / u.s)    
    sizes = np.maximum(1 + diff / 100, 1)
    sc = ax.scatter(
        l_rad[cut],
        b_rad[cut],
        s=sizes[cut],
        alpha=1,
        c=v_rad_gal[cut].to_value(u.km / u.s),
        cmap='coolwarm',
        # edgecolors="black",
        linewidths=0.1,
        rasterized=True,
        vmin=-400,
        vmax=400,
    )
    fig.colorbar(sc, ax=ax, label=r"$v_\mathrm{gal}\rm\ [km\ s^{-1}]$", fraction=0.02)

    ax.set_xlabel(r"$\mathrm{Galactic\ longitude}\ l$")
    ax.set_ylabel(r"$\mathrm{Galactic\ latitude}\ b$")
    ax.grid(True, alpha=0.3)

    tick_labels = [
        r"$150^\circ$", r"$120^\circ$", r"$90^\circ$",
        r"$60^\circ$", r"$30^\circ$", r"$0^\circ$",
        r"$330^\circ$", r"$300^\circ$", r"$270^\circ$",
        r"$240^\circ$", r"$210^\circ$",
    ]
    ax.set_xticklabels(tick_labels, fontsize=8)
    tick_labels = [r"$-75^\circ$", r"$-60^\circ$", r"$-45^\circ$", r"$-30^\circ$", r"$-15^\circ$", r"$0^\circ$", r"$15^\circ$", r"$30^\circ$", r"$45^\circ$", r"$60^\circ$", r"$75^\circ$"]
    ax.set_yticklabels(tick_labels, fontsize=8)

    if title: 
        fig.suptitle(title)

    plt.tight_layout()
    if show: plt.show()
    return fig, ax

def plot_galactic_velocity_lmc_rf(sky_coords, lmc_coord, cut):
    from matplotlib.colors import ListedColormap

    l_hvs, b_hvs = sky_coords.galactic.l, sky_coords.galactic.b

    # wrap longitude from [0, 360] to [-180, 180]
    l_wrapped = ((l_hvs.to_value(u.deg) + 180) % 360) - 180

    l_rad = np.radians(-l_wrapped)
    b_rad = np.radians(b_hvs.to_value(u.deg))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="mollweide")

    lon, lat, footprint = load_footprint_mollweide_grid("sdss_dr8_footprint.fits", plot_frame="galactic", map_frame="icrs")
    gray_cmap = ListedColormap(["0.85"])
    ax.pcolormesh(
        lon,
        lat,
        np.where(footprint > 0, 1.0, np.nan),
        shading="nearest",
        cmap=gray_cmap,
        alpha=1.0,
        rasterized=True,
        zorder=0,
    )

    # Radial velocity in the LMC rest frame
    sc_gal = sky_coords.transform_to("galactocentric")
    lmc_gal = lmc_coord.transform_to("galactocentric")

    # Relative position: LMC -> star
    dx = sc_gal.x - lmc_gal.x
    dy = sc_gal.y - lmc_gal.y
    dz = sc_gal.z - lmc_gal.z
    r_rel = np.sqrt(dx**2 + dy**2 + dz**2)

    # Relative velocity: star velocity - LMC velocity
    dvx = sc_gal.v_x - lmc_gal.v_x
    dvy = sc_gal.v_y - lmc_gal.v_y
    dvz = sc_gal.v_z - lmc_gal.v_z

    v_rad_lmc = (
        dvx * dx / r_rel
        + dvy * dy / r_rel
        + dvz * dz / r_rel
    ).to(u.km / u.s)

    sizes = 0.1 + 1 * np.abs(v_rad_lmc) / (2000 * u.km / u.s)

    sc = ax.scatter(
        l_rad[cut],
        b_rad[cut],
        s=sizes[cut].to_value(u.dimensionless_unscaled),
        alpha=1,
        c=v_rad_lmc[cut].to_value(u.km / u.s),
        cmap="coolwarm",
        linewidths=0.1,
        rasterized=True,
        vmin=-1000,
        vmax=1000,
    )
    fig.colorbar(
        sc,
        ax=ax,
        label=r"$v_{\rm rad,LMC}\rm\ [km\ s^{-1}]$",
        fraction=0.03,
    )

    ax.set_xlabel(r"$\mathrm{Galactic\ longitude}\ l$")
    ax.set_ylabel(r"$\mathrm{Galactic\ latitude}\ b$")
    ax.grid(True, alpha=0.3)

    tick_labels = [
        r"$150^\circ$", r"$120^\circ$", r"$90^\circ$",
        r"$60^\circ$", r"$30^\circ$", r"$0^\circ$",
        r"$330^\circ$", r"$300^\circ$", r"$270^\circ$",
        r"$240^\circ$", r"$210^\circ$",
    ]
    ax.set_xticklabels(tick_labels)
    tick_labels = [r"$-75^\circ$", r"$-60^\circ$", r"$-45^\circ$", r"$-30^\circ$", r"$-15^\circ$", r"$0^\circ$", r"$15^\circ$", r"$30^\circ$", r"$45^\circ$", r"$60^\circ$", r"$75^\circ$"]
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.show()