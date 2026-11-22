import tempfile
import unittest
from pathlib import Path

import control

import pidtune
from pid import Limits
from pidtune.loop import ki_that_holds
from plantid.plant import io_system, save


class PidTuneTest(unittest.TestCase):
    def test_load_plant_reads_identify_json(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plant.json"
            save(plant, path, model="całkujący + tarcie", kp=2.0, rmse=0.1)
            loaded = pidtune.load_plant(path)
        self.assertIsInstance(loaded, control.NonlinearIOSystem)
        self.assertEqual(loaded.params["kind"], "integrator")
        self.assertEqual(loaded.params["gain"], 0.8)
        self.assertEqual(loaded.params["fc"], 9.0)

    def test_tune_stays_inside_the_caller_limits(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0, 0.0),
            hold=1.0,
            dt=0.05,
            kp=(1.0, 4.0),
            ki=(0.0, 8.0),
            kd=(0.0,),
            encoder_step=None,
            allow_steady_state_error=False,
            generations=2,
            population=6,
            seed=1,
        )
        floor = ki_that_holds(plant, (0.0, 20.0, 0.0), Limits(), 0.05)
        self.assertGreaterEqual(tuned.gains.kp, 1.0)
        self.assertLessEqual(tuned.gains.kp, 4.0)
        self.assertGreaterEqual(tuned.gains.ki, floor)
        self.assertLessEqual(tuned.gains.ki, 8.0)
        self.assertEqual(tuned.gains.kd, 0.0)
        self.assertGreater(tuned.ise, 0.0)

    def test_tune_can_search_below_the_holding_gain(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0),
            hold=1.0,
            dt=0.05,
            kp=(2.0,),
            ki=(0.0, 1.0),
            kd=(0.0,),
            encoder_step=None,
            allow_steady_state_error=True,
            generations=1,
            population=4,
            seed=1,
        )
        self.assertLessEqual(tuned.gains.ki, 1.0)
        self.assertEqual(tuned.gains.kp, 2.0)

    def test_ki_is_not_raised_above_the_caller_limit(self):
        plant = io_system("inertial", gain=1.0, taus=(1.0,), fc=4.0)
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 90.0),
            hold=1.0,
            dt=0.015,
            kp=(10.0,),
            ki=(0.0, 20.0),
            kd=(0.0,),
            encoder_step=None,
            allow_steady_state_error=False,
        )
        self.assertEqual(tuned.gains.ki, 20.0)
        self.assertEqual(tuned.gains.kp, 10.0)

    def test_powell_prefers_the_gain_that_tracks(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 40.0, 0.0),
            hold=4.0,
            dt=0.05,
            kp=(4.0, 10.0),
            ki=(0.0,),
            kd=(0.0,),
            encoder_step=None,
            allow_steady_state_error=True,
            method="powell",
        )
        self.assertGreater(tuned.gains.kp, 4.5)
        self.assertEqual(tuned.gains.ki, 0.0)

    def test_powell_stays_inside_the_caller_limits(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0, 0.0),
            hold=1.0,
            dt=0.05,
            kp=(1.0, 4.0),
            ki=(0.0, 8.0),
            kd=(0.0,),
            encoder_step=None,
            allow_steady_state_error=False,
            method="powell",
        )
        floor = ki_that_holds(plant, (0.0, 20.0, 0.0), Limits(), 0.05)
        self.assertGreaterEqual(tuned.gains.kp, 1.0)
        self.assertLessEqual(tuned.gains.kp, 4.0)
        self.assertGreaterEqual(tuned.gains.ki, floor)
        self.assertLessEqual(tuned.gains.ki, 8.0)
        self.assertEqual(tuned.gains.kd, 0.0)

    def test_unknown_method_is_rejected(self):
        plant = io_system("integrator", gain=0.8, taus=(), fc=9.0)
        with self.assertRaises(ValueError):
            pidtune.tune(
                plant,
                levels=(0.0, 20.0),
                hold=1.0,
                kp=(1.0, 2.0),
                ki=(0.0, 1.0),
                method="pyomo",
            )


if __name__ == "__main__":
    unittest.main()
