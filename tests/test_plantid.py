import unittest

import control
import numpy as np
import polars as pl

from plantid import (
    ENCODER_STEP,
    LOG_STEPS,
    FirmwarePID,
    Gains,
    Limits,
    closed_loop,
    fit_model,
    inertial,
    integrator,
    search,
    simulate,
    tune,
)
from plantid.plant import Plant


class PlantIdentificationTest(unittest.TestCase):
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
        response = control.forced_response(plant.system, time, command)
        reference = np.squeeze(response.outputs)
        self.assertLess(np.max(np.abs(y - reference)), 0.02)

    def test_closed_loop_matches_feedback(self):
        plant = Plant("inertial", gain=3.0, taus=(0.4,), fc=0.0)
        kp = 2.0
        dt = 0.001
        time = np.arange(0.0, 4.0, dt)
        setpoint = np.ones_like(time)
        y = plant.simulate(time, setpoint, kp, 0.0)
        loop = closed_loop(plant.system, kp)
        reference = np.squeeze(control.forced_response(loop, time, setpoint).outputs)
        self.assertLess(np.max(np.abs(y - reference)), 0.05)

    def test_recovers_integrator_with_friction(self):
        true = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        time = np.arange(0.0, 16.0, 0.02)
        setpoint = np.where(time < 8.0, 40.0, -20.0)
        measured = true.simulate(time, setpoint, kp=1.0, y0=0.0)
        fitted = fit_model(time, setpoint, measured, integrator(friction=True), kp=1.0)
        self.assertAlmostEqual(fitted.gain, 0.8, delta=0.05)
        self.assertAlmostEqual(fitted.fc, 9.0, delta=0.3)
        self.assertLess(fitted.rmse, 0.05)

    def test_search_ranks_the_generating_model_first(self):
        true = Plant("inertial", gain=5.0, taus=(0.5, 0.1), fc=4.0)
        time = np.arange(0.0, 12.0, 0.02)
        setpoint = np.where(time < 6.0, 20.0, 0.0)
        measured = true.simulate(time, setpoint, kp=1.0, y0=0.0)
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
        pid = FirmwarePID(Gains(2.0, 10.0, 0.015), Limits(min_output=0.0))
        # First sample: D uses last_error = 0, so d = Kd * error / dt = 10.
        out = pid.step(10.0, 0.0, 0.015)
        self.assertAlmostEqual(out, 31.5)
        self.assertAlmostEqual(pid.acc, 10.0)
        out = pid.step(10.0, 0.0, 0.015)
        self.assertAlmostEqual(out, 23.0)
        self.assertAlmostEqual(pid.acc, 20.0)

    def test_dead_zone_min_output_and_anti_windup(self):
        pid = FirmwarePID(Gains(1.0, 0.0, 0.0), Limits(dead_zone=0.5, anti_windup=100.0))
        self.assertEqual(pid.step(0.1, 0.0, 0.015), 0.0)
        self.assertEqual(pid.acc, 0.0)

        lifted = FirmwarePID(Gains(0.01, 0.0, 0.0), Limits(min_output=5.0, max_output=100.0))
        self.assertEqual(lifted.step(1.0, 0.0, 0.015), 5.0)

        clamped = FirmwarePID(Gains(0.0, 1.0, 0.0), Limits(anti_windup=2.0))
        clamped.step(10.0, 0.0, 0.015)
        self.assertEqual(clamped.acc, 2.0)

    def test_dead_zone_parameter_blanks_one_count(self):
        plant = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = tune(
            plant,
            dt=0.05,
            hold=0.2,
            levels=(0.0, 30.0),
            kp=(4.0,),
            ki=(0.0,),
            kd=(0.0,),
            dead_zone=0.2,
            encoder_step=None,
            max_overshoot=100.0,
            max_final_error=30.0,
        )
        self.assertEqual(tuned.limits.dead_zone, 0.2)
        pid = FirmwarePID(Gains(8.0, 10.0, 0.0), tuned.limits)
        self.assertEqual(pid.step(ENCODER_STEP, 0.0, 0.015), 0.0)
        self.assertGreater(abs(pid.step(2 * ENCODER_STEP, 0.0, 0.015)), 0.0)

    def test_proportional_loop_stops_at_friction(self):
        plant = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        reference = np.full(400, 40.0)
        y, _ = simulate(plant, reference, Gains(1.0, 0.0, 0.0), dt=0.02, encoder_step=None)
        self.assertAlmostEqual(y[-1], 31.0, delta=0.5)

    def test_tune_takes_a_plant_and_scores_every_logged_step(self):
        plant = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = tune(
            plant,
            dt=0.05,
            hold=5.0,
            kp=(1.0,),
            ki=(0.0, 8.0),
            kd=(0.0,),
            encoder_step=None,
            we=1.0,
            max_overshoot=200.0,
            max_final_error=5.0,
        )
        self.assertEqual(tuned.fc, plant.fc)
        self.assertEqual(tuned.gains.kp, 1.0)
        self.assertEqual(tuned.gains.kd, 0.0)
        self.assertGreater(tuned.gains.ki, 1.0)
        self.assertLess(tuned.gains.ki, 8.0)
        self.assertTrue(tuned.meets_spec)
        label = f"Kp={tuned.gains.kp:g}, Ki={tuned.gains.ki:g}, Kd={tuned.gains.kd:g}"
        setpoints = tuned.steps.filter(pl.col("regulator") == label)["setpoint"].to_list()
        self.assertEqual(setpoints, list(LOG_STEPS))
        moves = [item for item in setpoints if item != 0.0]
        self.assertEqual(moves, [-30.0, -60.0, -90.0, 30.0, 60.0, 90.0])

    def test_tune_keeps_the_gain_that_tracks_the_setpoint(self):
        plant = Plant("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = tune(
            plant,
            dt=0.05,
            hold=4.0,
            levels=(0.0, 40.0, 0.0),
            kp=(4.0, 10.0),
            ki=(0.0,),
            kd=(0.0,),
            encoder_step=None,
            max_overshoot=100.0,
            max_final_error=20.0,
        )
        self.assertGreater(tuned.gains.kp, 4.5)
        self.assertEqual(tuned.gains.ki, 0.0)

    def test_tune_rejects_a_lower_ise_that_stops_short_of_the_setpoint(self):
        plant = Plant("inertial", gain=321.2, taus=(427.0, 0.07, 0.07), fc=10.16)
        tuned = tune(
            plant,
            dt=0.015,
            hold=6.0,
            levels=(0.0, 90.0, 0.0),
            kp=(3.0, 12.0),
            ki=(0.5, 7.0),
            kd=(0.0, 0.5),
            dead_zone=0.1,
            encoder_step=None,
            we=1.0,
            max_overshoot=20.0,
            max_final_error=0.2,
        )
        self.assertGreater(tuned.gains.ki, 5.0)
        i_max = tuned.limits.anti_windup * tuned.dt * tuned.gains.ki
        self.assertGreaterEqual(i_max, plant.fc + 90.0 / plant.gain)
        worst = max(step.abs_final_deg for step in tuned.score.steps)
        self.assertLess(worst, 0.2)

    def test_tune_does_not_raise_ki_above_the_caller_limit(self):
        plant = Plant("inertial", gain=1.0, taus=(1.0,), fc=4.0)
        tuned = tune(
            plant,
            dt=0.015,
            hold=1.0,
            levels=(0.0, 90.0),
            kp=(10.0,),
            ki=(0.0, 20.0),
            kd=(0.0,),
            encoder_step=None,
            max_overshoot=1000.0,
            max_final_error=100.0,
        )
        self.assertEqual(tuned.gains.ki, 20.0)

    def test_tune_can_leave_a_steady_state_error_when_asked(self):
        plant = Plant("inertial", gain=321.2, taus=(427.0, 0.07, 0.07), fc=10.16)
        tuned = tune(
            plant,
            dt=0.015,
            hold=6.0,
            levels=(0.0, 90.0, 0.0),
            kp=(3.0, 12.0),
            ki=(0.5, 7.0),
            kd=(0.0,),
            dead_zone=0.1,
            encoder_step=None,
            allow_steady_state_error=True,
            max_overshoot=100.0,
            max_final_error=0.2,
        )
        self.assertLess(tuned.gains.ki, 2.0)


if __name__ == "__main__":
    unittest.main()
