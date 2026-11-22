import tempfile
import unittest
from pathlib import Path

import control
import numpy as np

from pid import PID, Gains, Limits
from pidtune.loop import simulate
from plantid import Fit, Log, Model, Plant


class PlantIdentificationTest(unittest.TestCase):
    def test_plant_json_roundtrip_keeps_the_dynamics(self):
        plant = Plant("inertial", gain=2.5, taus=(1.5, 0.2), fc=4.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plant.json"
            plant.save(path, model="inercyjny 2 + tarcie", kp=1.0, rmse=0.2)
            loaded = Plant.load(path)
        self.assertIsInstance(loaded, Plant)
        self.assertEqual(loaded.kind, plant.kind)
        self.assertEqual(loaded.gain, plant.gain)
        self.assertEqual(loaded.taus, plant.taus)
        self.assertEqual(loaded.fc, plant.fc)

    def test_open_loop_matches_python_control(self):
        plant = Plant("inertial", gain=4.0, taus=(0.3,), fc=0.0)
        dt = 0.001
        time = np.arange(0.0, 3.0, dt)
        command = np.ones_like(time)
        state = plant.initial_state(0.0)
        y = np.empty_like(time)
        y[0] = 0.0
        for i in range(len(time) - 1):
            state, y[i + 1] = plant.step(state, command[i], dt)
        response = control.forced_response(plant.linear, time, command)
        reference = np.squeeze(response.outputs)
        self.assertLess(np.max(np.abs(y - reference)), 0.02)

    def test_closed_loop_matches_feedback(self):
        plant = Plant("inertial", gain=3.0, taus=(0.4,), fc=0.0)
        kp = 2.0
        dt = 0.001
        time = np.arange(0.0, 4.0, dt)
        setpoint = np.ones_like(time)
        y = plant.proportional_response(time, setpoint, kp, 0.0)
        loop = plant.closed_loop(kp)
        reference = np.squeeze(control.forced_response(loop, time, setpoint).outputs)
        self.assertLess(np.max(np.abs(y - reference)), 0.05)

    def test_recovers_integrator_with_friction(self):
        true = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        time = np.arange(0.0, 16.0, 0.02)
        setpoint = np.where(time < 8.0, 40.0, -20.0)
        measured = true.proportional_response(time, setpoint, 1.0, 0.0)
        fitted = Model.integrator(friction=True).fit(time, setpoint, measured, kp=1.0)
        self.assertAlmostEqual(fitted.gain, 0.8, delta=0.05)
        self.assertAlmostEqual(fitted.fc, 9.0, delta=0.3)
        self.assertLess(fitted.rmse, 0.05)

    def test_search_ranks_the_generating_model_first(self):
        true = Plant("inertial", gain=5.0, taus=(0.5, 0.1), fc=4.0)
        time = np.arange(0.0, 12.0, 0.02)
        setpoint = np.where(time < 6.0, 20.0, 0.0)
        measured = true.proportional_response(time, setpoint, 1.0, 0.0)
        found = Fit.search(
            time,
            setpoint,
            measured,
            [
                Model.inertial(1, friction=True),
                Model.inertial(2, friction=True),
                Model.integrator(friction=True),
            ],
            kp=1.0,
        )
        self.assertEqual(found.best.model.name, "inercyjny 2 + tarcie")
        self.assertLess(found.best.rmse, 0.2)
        self.assertEqual(found.kp, 1.0)
        self.assertEqual(found.best.kp, 1.0)

    def test_loop_gain_reads_recorded_p(self):
        error = np.array([0.0, 4.0, -2.0, 10.0])
        p_term = 2.0 * error
        self.assertEqual(Log.gain(error, p_term), 2.0)


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
        count = 360.0 / 2048.0
        self.assertEqual(pid.step(count, 0.0, 0.015), 0.0)
        self.assertGreater(abs(pid.step(2 * count, 0.0, 0.015)), 0.0)

    def test_simulate_dead_zone_blanks_the_command(self):
        plant = Plant("integrator", gain=0.8, taus=(), fc=0.0)
        reference = np.full(5, 10.0)
        _y, command = simulate(
            plant,
            reference,
            Gains(1.0, 0.0, 0.0),
            dt=0.05,
            dead_zone=100.0,
        )
        self.assertTrue(np.all(command == 0.0))

    def test_proportional_loop_stops_at_friction(self):
        plant = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        reference = np.full(400, 40.0)
        y, _ = simulate(plant, reference, Gains(1.0, 0.0, 0.0), dt=0.02)
        self.assertAlmostEqual(y[-1], 31.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
