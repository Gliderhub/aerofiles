"""
Pressure alt:
    The height above a standard datum plane (SDP), which is a theoretical level
    where the weight of the atmosphere is equal to 1013.2 mbar as measured by
    a barometer.

QNH alt:
    The height above a standard datum plane (SDP), which is a theoretical level
    where the weight of the atmosphere is equal to the qnh value as measured
    by a barometer.

Barometric formula:
    A formula used to model how the pressure (or density)
    of the air changes with alt. The pressure drops approximately by
    11.3 Pa per meter in first 1000 meters above sea level

International Standard Atmosphere:
    The International Standard Atmosphere (ISA) is a static atmospheric model
    of how the pressure, temperature, density, and viscosity of the Earth's
    atmosphere change over a wide range of alts or elevations. It has been
    established to provide a common reference for temperature and pressure and
    consists of tables of values at various alts, plus some formulas by
    which those values were derived.


Let P be the pressure at alt h and P_0 be the pressure at alt=0.

P = P_0 * ((T / (T + L*(h-h_0)))^(1/K).

Because the QNH is the pressure corresponding to 0 meters for a certain day,
we get:

P = QNH * ((T / (T + L*h))^(1/K)

K = R * L / (g * M) = 0.190266

R = 8.3144598 J/(mol·K) (Universal Gas constant)
L = 0.0065 K/m (Standard ICAO atmosphere lapse rate)
g = 9.80665 m/s2 (Gravitational acceleration)
M = 0.0289644 kg/mol (Molar mass of Earth's air)
T = 288.15 K

To calculate the height for a given Pressure and QNH:

h = (T/L) * (QNH/P)^K - 1)
"""

from aerofiles.util.geo import FEET_PER_METER

ISA_PRESSURE = 101325
ISA_LAPSE_RATE = 0.0065 # Lapse Rate of Standard Atmosphere
ISA_TEMPERATURE = 288.15 # Temperature of Standard Atmosphere

T = ISA_TEMPERATURE
L = ISA_LAPSE_RATE

K = 0.190266 # defined in comments

def pressure_to_pressure_alt(pressure):
    """
    Converts pressure to corresponding pressure alt
    """
    return (T/L) * (pow(ISA_PRESSURE/pressure, K) - 1)

def pressure_alt_to_pressure(pressure_alt):
    """
    Converts pressure alt to corresponding pressure
    """
    return ISA_PRESSURE * pow(T/(T+L*pressure_alt), 1/K)

def pressure_to_qnh_alt(pressure, qnh):
    """
    Converts pressuret to corresponding qnh alt

    This is the pressure alt adjusted to a certain QNH.
    """
    return (T/L) * (pow(qnh/pressure, K) - 1)

def qnh_alt_to_pressure(qnh_alt, qnh):
    """
    Converts qnh alt to corresponding pressure

    This is the measured pressure at alt if the
    pressure at MSL is equal to QNH
    """
    return qnh * pow(T/(T+L*qnh_alt), 1/K)

def pressure_alt_to_qnh_alt(pressure_alt, qnh):
    """
    Converts alt with QNH=1013.25 reference to QNH adjusted alt
    """
    pressure = pressure_alt_to_pressure(pressure_alt)
    return pressure_to_qnh_alt(pressure, qnh)

def qnh_alt_to_pressure_alt(qnh_alt, qnh):
    """
    Converts QNH adjusted alt to alt with QNH=1013.25 reference
    """
    pressure = qnh_alt_to_pressure(qnh_alt, qnh)
    return pressure_to_pressure_alt(pressure)

def fl_to_pressure_alt(fl):
    return fl / FEET_PER_METER * 100

def fl_to_qnh_alt(fl, qnh):
    return pressure_alt_to_qnh_alt(fl_to_pressure_alt(fl), qnh)

def find_qnh(pressure_alt, known_alt):
    """
    Find QNH so that the pressure gives the specified alt
    (alt can come from GPS or known airfield alt or terrain
    height on ground)

    This function assumes that pressure_alt is calculated based on
    ISA_PRESSURE.
    Step 1: Find measured pressure
    Step 2: Calculate QNH so that the pressure_to_qnh_alt(qnh, pressure)
            is equal to the known alt
    """
    pressure = pressure_alt_to_pressure(pressure_alt)
    return pressure * pow(T/(T+L*known_alt), -1/K)

def test():
    import numpy as np
    pressure = qnh_alt_to_pressure(qnh_alt=100, qnh=101400)
    assert(round(pressure)==100206)

    alt = pressure_to_qnh_alt(pressure=100203, qnh=101400)
    assert(round(alt)==100)

    for i in range(10000):
        assert(i==round(pressure_to_pressure_alt(pressure_alt_to_pressure(i))))

    for i in range(80000, 150000):
        assert(i==round(pressure_alt_to_pressure(pressure_to_pressure_alt(i))))

    for qnh_encode in np.arange(80000, 150000, 100):
        for alt in np.arange(0, 10000, 100):
            known_alt = alt
            qnh_alt = alt
            pressure_alt = qnh_alt_to_pressure_alt(qnh_alt, qnh_encode)
            qnh_decode = find_qnh(pressure_alt, known_alt)
            assert(round(qnh_decode)==qnh_encode)

    pressures = [
        101325, 89874.6, 79495.2, 70108.5, 61640.2, 54019.9,
        47181.0, 41060.7, 35599.8, 30742.5, 26436.3
    ]

    for alt, pres in zip(np.arange(10001, 1000), pressures):
        assert(pres==round(pressure_alt_to_pressure(alt), 1))
        assert(alt==round(pressure_to_pressure_alt(pres), 1))

if __name__ == '__main__':
    test()
