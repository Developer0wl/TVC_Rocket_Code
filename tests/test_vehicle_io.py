import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tvc.motor import Motor
from tvc.params import VehicleParams
from tvc.sim import Sim
from tvc.vehicle_io import VehicleFileError, load_vehicle, params_from_dict

TEMPLATE = Path(__file__).resolve().parents[1] / "vehicle" / "vehicle_template.json"


def template():
    return json.loads(TEMPLATE.read_text())


def test_template_reproduces_builtin_mass_properties():
    loaded, builtin = Motor(load_vehicle(TEMPLATE)), Motor(VehicleParams())
    for t in (0.0, 0.5, 1.5, 3.0, 10.0):
        m1, d1, i1 = loaded.mass_properties(t)
        m2, d2, i2 = builtin.mass_properties(t)
        assert m1 == pytest.approx(m2, rel=1e-4)
        assert d1 == pytest.approx(d2, rel=2e-3)
        np.testing.assert_allclose(i1, i2, rtol=2e-2)


def test_table_endpoints_and_interpolation():
    p = load_vehicle(TEMPLATE)
    motor = Motor(p)
    (m0, x0, _, _), (m1, x1, _, _) = p.mass_table
    assert motor.mass_properties(0.0)[0] == pytest.approx(m0)
    assert motor.mass_properties(0.0)[1] == pytest.approx(x0)
    assert motor.mass_properties(10.0)[0] == pytest.approx(m1)
    assert motor.mass_properties(10.0)[1] == pytest.approx(x1)
    mid = motor.mass_properties(1.5)[0]
    assert m1 < mid < m0


def test_cad_inertia_changes_dynamics():
    data = template()
    data["mass_properties"]["ignition"]["Iyy_kgm2"] *= 2
    data["mass_properties"]["burnout"]["Iyy_kgm2"] *= 2
    heavy = Motor(params_from_dict(data)).mass_properties(0.5)[2][1]
    base = Motor(load_vehicle(TEMPLATE)).mass_properties(0.5)[2][1]
    assert heavy == pytest.approx(2 * base)


def test_sim_runs_on_loaded_vehicle():
    sim = Sim(load_vehicle(TEMPLATE), dt=1e-3)
    while sim.on_rail and sim.t < 2.0:
        sim.step((0.0, 0.0))
    assert not sim.on_rail and sim.rail_exit_speed > 1.0


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d["dimensions"].update(length_m=600.0), "mm instead of m"),
        (lambda d: d["mass_properties"]["ignition"].update(mass_kg=1000.0), "g instead of kg"),
        (lambda d: d["mass_properties"]["burnout"].update(x_cm_m=2.0), "within 0..length"),
        (lambda d: d["mass_properties"]["burnout"].update(mass_kg=1.2), "less than ignition"),
        (lambda d: d["motor"].update(propellant_mass_kg=0.2), "disagrees with propellant"),
        (lambda d: d["motor"].update(thrust_N=[0.0, 30.0]), "matching time/thrust"),
        (lambda d: d["mass_properties"]["burnout"].update(Iyy_kgm2=0.0001), "should exceed Ixx"),
        (lambda d: d.pop("tvc"), "missing 'tvc'"),
    ],
)
def test_validation_catches_bad_cad_numbers(mutate, match):
    data = copy.deepcopy(template())
    mutate(data)
    with pytest.raises(VehicleFileError, match=match):
        params_from_dict(data)
