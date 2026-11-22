import tempfile
import unittest
from pathlib import Path

import control
import numpy as np

from pid import ENCODER_STEP, PID, Gains, Limits
from pidtune.loop import simulate
from plantid import (
    closed_loop,
    fit_model,
    inertial,
    integrator,
    search,
)
from plantid.plant import initial_state, io_system, load, proportional_response, save, step


class PlantIdentificationTest(unittest.TestCase):
    def test_plant_json_roundtrip_keeps_the_dynamics(self):
        plant = io_system("inertial", gain=2.5, taus=(1.5, 0.2), fc=4.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plant.json"
            save(plant, path, model="inercyjny 2 + tarcie", kp=1.0, rmse=0.2)
            loaded = load(path)
        self.assertIsInstance(loaded, control.NonlinearIOSystem)
        self.assertEqual(loaded.params["kind"], plant.params["kind"])
        self.assertEqual(loaded.params["gain"], plant.params["gain"])
        self.assertEqual(loaded.params["taus"], plant.params["taus"])
        self.assertEqual(loaded.params["fc"], plant.params["fc"])

    def test_open_loop_matches_python_control(self):
        plant = io_system("inertial", gain=4.0, taus=(0.3,), fc=0.0)
        dt = 0.001
        time = np.arange(0.0, 3.0, dt)
        command = np.ones_like(time)
        state = initial_state(plant, 0.0)
        y = np.empty_like(time)
        y[0] = 0.0
        for i in range(len(time) - 1):
            state, y[i + 1] = step(plant, state, command[i], dt)
        response = control.forced_response(plant.linear, time, command)
        reference = np.squeeze(response.outputs)
        self.assertLess(np.max(np.abs(y - reference)), 0.02)

    def test_closed_loop_matches_feedback(self):
        plant = io_system("inertial", gain=3.0, taus=(0.4,), fc=0.0)
        kp = 2.0
        dt = 0.001
        time = np.arange(0.0, 4.0, dt)
        setpoint = np.ones_like(time)
        y = proportional_response(plant, time, setpoint, kp, 0.0)
        loop = closed_loop(plant.linear, kp)
        reference = np.squeeze(control.forced_response(loop, time, setpoint).outputs)
        self.assertLess(np.max(np.abs(y - reference)), 0.05)

    def test_recovers_integrator_with_friction(self):
        true = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        time = np.arange(0.0, 16.0, 0.02)
        setpoint = np.where(time < 8.0, 40.0, -20.0)
        measured = proportional_response(true, time, setpoint, 1.0, 0.0)
        fitted = fit_model(time, setpoint, measured, integrator(friction=True), kp=1.0)
        self.assertAlmostEqual(fitted.gain, 0.8, delta=0.05)
        self.assertAlmostEqual(fitted.fc, 9.0, delta=0.3)
        self.assertLess(fitted.rmse, 0.05)

    def test_search_ranks_the_generating_model_first(self):
        true = io_system("inertial", gain=5.0, taus=(0.5, 0.1), fc=4.0)
        time = np.arange(0.0, 12.0, 0.02)
        setpoint = np.where(time < 6.0, 20.0, 0.0)
        measured = proportional_response(true, time, setpoint, 1.0, 0.0)
        found = search(
            time,
            setpoint,
            measured,
            [inertial(1, friction=True), inertial(2, friction=True), integrator(friction=True)],
            kp=1.0,
        )
        self.assertEqual(found.best.model.name, "inercyjny 2 + tarcie")
        self.assertLess(found.best.rmse, 0.2)
        self.assertEqual(found.kp, 1.0)
        self.assertEqual(found.best.kp, 1.0)

    def test_loop_gain_reads_recorded_p(self):
        from plantid import loop_gain

        error = np.array([0.0, 4.0, -2.0, 10.0])
        p_term = 2.0 * error
        self.assertEqual(loop_gain(error, p_term), 2.0)


class FirmwarePidTest(unittest.TestCase):
    def test_step_matches_firmware_law(self):
        pid = PID(Gains(2.0, 10.0, 0.015), Limits(min_output=0.0))
        # First sample: D uses last_error = 0, so d = Kd * error / dt = 10.
        out = pid.step(10.0, 0.0, 0.015)
        self.assertAlmostEqual(out, 31.5)
        self.assertAlmostEqual(pid.acc, 10.0)
        out = pid.step(10.0, 0.0, 0.015)
        self.assertAlmostEqual(out, 23.0)
        self.assertAlmostEqual(pid.acc, 20.0)

    def test_dead_zone_min_output_and_anti_windup(self):
        pid = PID(Gains(1.0, 0.0, 0.0), Limits(dead_zone=0.5, anti_windup=100.0))
        self.assertEqual(pid.step(0.1, 0.0, 0.015), 0.0)
        self.assertEqual(pid.acc, 0.0)

        lifted = PID(Gains(0.01, 0.0, 0.0), Limits(min_output=5.0, max_output=100.0))
        self.assertEqual(lifted.step(1.0, 0.0, 0.015), 5.0)

        clamped = PID(Gains(0.0, 1.0, 0.0), Limits(anti_windup=2.0))
        clamped.step(10.0, 0.0, 0.015)
        self.assertEqual(clamped.acc, 2.0)

    def test_dead_zone_blanks_one_encoder_count(self):
        pid = PID(Gains(8.0, 10.0, 0.0), Limits(dead_zone=0.2))
        self.assertEqual(pid.step(ENCODER_STEP, 0.0, 0.015), 0.0)
        self.assertGreater(abs(pid.step(2 * ENCODER_STEP, 0.0, 0.015)), 0.0)

    def test_simulate_dead_zone_blanks_the_command(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=0.0)
        reference = np.full(5, 10.0)
        _y, command = simulate(
            plant,
            reference,
            Gains(1.0, 0.0, 0.0),
            dt=0.05,
            dead_zone=100.0,
            encoder_step=None,
        )
        self.assertTrue(np.all(command == 0.0))

    def test_proportional_loop_stops_at_friction(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        reference = np.full(400, 40.0)
        y, _ = simulate(plant, reference, Gains(1.0, 0.0, 0.0), dt=0.02, encoder_step=None)
        self.assertAlmostEqual(y[-1], 31.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
