import numpy as np
import healpy as hp
import randomsdss

n_random = 1_000_000
nside = 256
nest = True

outfile = "sdss_dr8_footprint.fits"

ra, dec = randomsdss.sky_random(dr="DR8", size=n_random)

theta = np.radians(90.0 - np.asarray(dec))
phi = np.radians(np.asarray(ra))

pix = hp.ang2pix(nside, theta, phi, nest=nest)

mask = np.zeros(hp.nside2npix(nside), dtype=np.int16)
mask[np.unique(pix)] = 1

hp.write_map(
    outfile,
    mask,
    nest=nest,
    coord="C",   # equatorial / ICRS-ish
    overwrite=True,
    dtype=np.int16,
)