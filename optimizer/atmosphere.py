"""International Standard Atmosphere (0-11 km) and freestream quantities."""
from __future__ import annotations

import math
from dataclasses import dataclass

R_AIR = 287.058
GAMMA = 1.4
T0 = 288.15
P0 = 101325.0
LAPSE = 0.0065
G0 = 9.80665


@dataclass(frozen=True)
class FlowConditions:
    altitude_m: float
    temperature: float      # K
    pressure: float         # Pa
    density: float          # kg/m^3
    dynamic_viscosity: float  # Pa s
    kinematic_viscosity: float  # m^2/s
    speed_of_sound: float   # m/s
    speed: float            # m/s
    aoa_deg: float

    @property
    def mach(self) -> float:
        return self.speed / self.speed_of_sound

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.density * self.speed ** 2

    def reynolds(self, length: float) -> float:
        return self.speed * length / self.kinematic_viscosity

    def flow_direction(self):
        """Unit freestream vector (x aft, y span, z up) for the given AoA."""
        a = math.radians(self.aoa_deg)
        return (math.cos(a), 0.0, math.sin(a))

    def lift_direction(self):
        a = math.radians(self.aoa_deg)
        return (-math.sin(a), 0.0, math.cos(a))


def isa(altitude_m: float):
    if not 0.0 <= altitude_m <= 11000.0:
        raise ValueError("ISA implementation covers 0-11000 m")
    t = T0 - LAPSE * altitude_m
    p = P0 * (t / T0) ** (G0 / (R_AIR * LAPSE))
    rho = p / (R_AIR * t)
    mu = 1.458e-6 * t ** 1.5 / (t + 110.4)          # Sutherland
    a = math.sqrt(GAMMA * R_AIR * t)
    return t, p, rho, mu, a


def flow_conditions(altitude_m: float, speed: float, aoa_deg: float) -> FlowConditions:
    t, p, rho, mu, a = isa(altitude_m)
    return FlowConditions(
        altitude_m=altitude_m,
        temperature=t,
        pressure=p,
        density=rho,
        dynamic_viscosity=mu,
        kinematic_viscosity=mu / rho,
        speed_of_sound=a,
        speed=speed,
        aoa_deg=aoa_deg,
    )
