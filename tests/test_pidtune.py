import tempfile
import unittest
from pathlib import Path

import pidtune
from plantid.plant import Plant


class PidTuneTest(unittest.TestCase):
    def test_load_plant_reads_identify_json(self):
        plant = Plant("integrator", gain=0.8, taus=())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plant.json"
            plant.save(path, model="integrator", kp=2.0, rmse=0.1)
            loaded = pidtune.load_plant(path)
        self.assertIsInstance(loaded, Plant)
        self.assertEqual(loaded.kind, "integrator")
        self.assertEqual(loaded.gain, 0.8)
        self.assertEqual(loaded.taus, ())

    def test_tune_stays_inside_the_caller_limits(self):
        plant = Plant("integrator", gain=0.8, taus=())
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0, 0.0),
            hold=1.0,
            dt=0.05,
            kp=(1.0, 4.0),
            ki=(0.0, 8.0),
            kd=(0.0,),
            generations=2,
            population=6,
            seed=1,
        )
        self.assertGreaterEqual(tuned.gains.kp, 1.0)
        self.assertLessEqual(tuned.gains.kp, 4.0)
        self.assertGreaterEqual(tuned.gains.ki, 0.0)
        self.assertLessEqual(tuned.gains.ki, 8.0)
        self.assertEqual(tuned.gains.kd, 0.0)
        self.assertGreater(tuned.ise, 0.0)

    def test_fixed_gains_are_kept(self):
        plant = Plant("integrator", gain=0.8, taus=())
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0),
            hold=1.0,
            dt=0.05,
            kp=(2.0,),
            ki=(0.0, 1.0),
            kd=(0.0,),
            generations=1,
            population=4,
            seed=1,
        )
        self.assertLessEqual(tuned.gains.ki, 1.0)
        self.assertEqual(tuned.gains.kp, 2.0)

    def test_powell_prefers_the_gain_that_tracks(self):
        plant = Plant("integrator", gain=0.8, taus=())
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 40.0, 0.0),
            hold=4.0,
            dt=0.05,
            kp=(4.0, 10.0),
            ki=(0.0,),
            kd=(0.0,),
            method="powell",
        )
        self.assertGreater(tuned.gains.kp, 4.5)
        self.assertEqual(tuned.gains.ki, 0.0)

    def test_powell_stays_inside_the_caller_limits(self):
        plant = Plant("integrator", gain=0.8, taus=())
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0, 0.0),
            hold=1.0,
            dt=0.05,
            kp=(1.0, 4.0),
            ki=(0.0, 8.0),
            kd=(0.0,),
            method="powell",
        )
        self.assertGreaterEqual(tuned.gains.kp, 1.0)
        self.assertLessEqual(tuned.gains.kp, 4.0)
        self.assertGreaterEqual(tuned.gains.ki, 0.0)
        self.assertLessEqual(tuned.gains.ki, 8.0)
        self.assertEqual(tuned.gains.kd, 0.0)

    def test_particle_swarm_stays_inside_the_caller_limits(self):
        plant = Plant("integrator", gain=0.8, taus=())
        tuned = pidtune.tune(
            plant,
            levels=(0.0, 20.0, 0.0),
            hold=1.0,
            dt=0.05,
            kp=(1.0, 4.0),
            ki=(0.0, 8.0),
            kd=(0.0,),
            method="pso",
            generations=2,
            population=4,
            seed=1,
        )
        self.assertGreaterEqual(tuned.gains.kp, 1.0)
        self.assertLessEqual(tuned.gains.kp, 4.0)
        self.assertGreaterEqual(tuned.gains.ki, 0.0)
        self.assertLessEqual(tuned.gains.ki, 8.0)
        self.assertEqual(tuned.gains.kd, 0.0)
        self.assertGreater(tuned.ise, 0.0)

    def test_unknown_method_is_rejected(self):
        plant = Plant("integrator", gain=0.8, taus=())
        with self.assertRaises(ValueError):
            pidtune.tune(
                plant,
                levels=(0.0, 20.0),
                hold=1.0,
                dt=0.05,
                kp=(1.0, 2.0),
                ki=(0.0, 1.0),
                method="pyomo",
            )


if __name__ == "__main__":
    unittest.main()
