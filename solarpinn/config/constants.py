"""
Physical and mathematical constants used throughout SolarPINN.

All values are given in SI units unless otherwise stated.
"""

from __future__ import annotations

import math

#Mathematical constants
PI: float = math.pi
TAU: float = math.tau
E: float = math.e

#Fudamental Physical Constants (SI)
SPEED_OF_LIGHT: float = 299_792_458.0              # m/s
GRAVITATIONAL_CONSTANT: float = 6.67430e-11        # m^3 kg^-1 s^-2
PLANCK_CONSTANT: float = 6.62607015e-34            # J s
REDUCED_PLANCK_CONSTANT: float = 1.054571817e-34   # J s
ELEMENTARY_CHARGE: float = 1.602176634e-19         # C
BOLTZMANN_CONSTANT: float = 1.380649e-23           # J/K
AVOGADRO_CONSTANT: float = 6.02214076e23           # mol^-1

#Electromagnetic Constants
VACUUM_PERMITTIVITY: float = 8.8541878128e-12      # F/m
VACUUM_PERMEABILITY: float = 1.25663706212e-6      # N/A^2

#Astronomical Constants
ASTRONOMICAL_UNIT: float = 1.495978707e11          # m
PARSEC: float = 3.085677581491367e16              # m
LIGHT_YEAR: float = 9.4607304725808e15            # m

#Solar constants
SOLAR_MASS: float = 1.98847e30                     # kg
SOLAR_RADIUS: float = 6.957e8                      # m
SOLAR_LUMINOSITY: float = 3.828e26                 # W
SOLAR_EFFECTIVE_TEMPERATURE: float = 5772.0        # K
SOLAR_SURFACE_GRAVITY: float = 274.0               # m/s^2

#Solar Rotation
SOLAR_ROTATION_PERIOD_DAYS: float = 27.2753
SOLAR_ROTATION_PERIOD_SECONDS: float = (
    SOLAR_ROTATION_PERIOD_DAYS * 24.0 * 3600.0
)

#Unit Conversions
TESLA_TO_GAUSS: float = 1.0e4
GAUSS_TO_TESLA: float = 1.0e-4

KM_TO_M: float = 1.0e3
M_TO_KM: float = 1.0e-3

DAY_TO_SECONDS: float = 86400.0
HOUR_TO_SECONDS: float = 3600.0

DEG_TO_RAD: float = PI / 180.0
RAD_TO_DEG: float = 180.0 / PI

