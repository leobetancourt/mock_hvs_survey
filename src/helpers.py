import numpy as np
import astropy.units as u

def random_dir_on_sphere():
    """
    Sample a random direction on a sphere.
    Returns theta (polar angle) and phi (azimuthal angle).
    """
    theta = np.arccos(2 * np.random.rand() - 1)  # polar angle
    phi = 2 * np.pi * np.random.rand()  # azimuthal angle
    return theta * u.rad, phi * u.rad

def spherical_to_cylindrical(r, theta, phi, vr, vtheta, vphi):
    """
    Convert spherical coordinates to cylindrical coordinates.
    """
    R = r * np.sin(theta)
    z = r * np.cos(theta)
    vR = vr * np.sin(theta) + vtheta * np.cos(theta)
    vz = vr * np.cos(theta) - vtheta * np.sin(theta)
    vphi_cyl = vphi
    return R, phi, z, vR, vz, vphi_cyl

def absolute_to_apparent_magnitude(m_abs, distance):
    """Convert absolute magnitude to apparent magnitude given a distance in parsecs."""
    return m_abs + 5 * np.log10(distance / 10.0)

def v_esc(R_kpc):
    """
    Approximate Galactic escape velocity as a function of Galactocentric radius R (in kpc).
    """
    return (
        624.9
        - 9.41543 * R_kpc
        + 0.134835346 * R_kpc**2
        - 1.292640e-3 * R_kpc**3
        + 6.5435315e-6 * R_kpc**4
        - 1.3312833e-8 * R_kpc**5
    ) * u.km / u.s

def convert_galactic_to_mollweide(l, b):
    l_wrapped = ((l.to_value(u.deg) + 180) % 360) - 180
    l_rad = np.radians(-l_wrapped)
    b_rad = np.radians(b.to_value(u.deg))
    return l_rad, b_rad

def angle_between(v1, v2):
    """
    Calculate the angle between two vectors in radians.
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # clip dot product to handle floating-point inaccuracies
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    # angle in radians
    angle_rad = np.arccos(dot_product)
    return angle_rad

def sample_power_law(x_min, x_max, alpha, size=1):
    """
    Sample from a power-law distribution p(x) ∝ x^alpha
    over the interval [x_min, x_max].
    """

    u = np.random.uniform(0, 1, size)

    if np.isclose(alpha, -1):
        # Special case: p(x) ∝ x^-1
        return x_min * (x_max / x_min) ** u

    exponent = alpha + 1

    return (
        x_min**exponent
        + u * (x_max**exponent - x_min**exponent)
    ) ** (1 / exponent)

def R_Eggleton(q, a_bin):
    """
    Calculate the Roche lobe radius using Eggleton's approximation.
    q: mass ratio (M_secondary / M_primary)
    a_bin: binary separation
    """
    return a_bin * 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))

def roche_lobe_radius_fraction(q):
    """
    Calculate Eggleton's Roche lobe radius fraction for a mass ratio q.
    """
    return 0.49 * q**(2 / 3) / (
        0.6 * q**(2 / 3) + np.log(1 + q**(1 / 3))
    )

def main_sequence_radius(m):
    # m in Msun, returns radius in Rsun; rough MS fit
    return np.where(m < 1 * u.Msun, (m / (1 * u.Msun))**0.8, (m / (1 * u.Msun))**0.57) * (1 * u.Rsun)