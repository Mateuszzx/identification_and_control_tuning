import tempfile
import unittest
from pathlib import Path

from plantid.config import load_config, repository_root, resolve_config


class ConfigTest(unittest.TestCase):
    def test_repo_config_saves_the_plant_in_model(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "config" / "config.yaml", root=root)
        self.assertEqual(cfg.identify.out, (root / "model" / "plant.json").resolve())
        self.assertEqual(cfg.tune.plant, cfg.identify.out)
        self.assertTrue(cfg.tune.levels)
        self.assertEqual(len(cfg.tune.kp), 2)

    def test_relative_paths_and_a_fixed_gain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.yaml").write_text(
                "\n".join(
                    [
                        "identify:",
                        "  log: data/log.csv",
                        "  out: model/plant.json",
                        "tune:",
                        "  plant: model/plant.json",
                        "  dt: 0.015",
                        "  dead_zone: 0.1",
                        "  hold: 6",
                        "  levels: [0, 90, 0]",
                        "  kp: [1, 10]",
                        "  ki: 3",
                        "  kd: [0]",
                        "  we: 1",
                        "  allow_steady_state_error: false",
                        "  generations: 4",
                        "  population: 6",
                        "",
                    ]
                )
            )
            cfg = load_config(root / "config.yaml")
        self.assertEqual(cfg.identify.log, (root / "data" / "log.csv").resolve())
        self.assertEqual(cfg.tune.ki, (3.0, 3.0))
        self.assertEqual(cfg.tune.kd, (0.0, 0.0))
        self.assertFalse(cfg.tune.allow_steady_state_error)
        self.assertEqual(cfg.tune.method, "pygad")
        self.assertEqual(cfg.tune.levels, (0.0, 90.0, 0.0))

    def test_unknown_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.yaml").write_text(
                "\n".join(
                    [
                        "identify:",
                        "  log: data/log.csv",
                        "  out: model/plant.json",
                        "tune:",
                        "  plant: model/plant.json",
                        "  dt: 0.015",
                        "  dead_zone: 0.1",
                        "  hold: 6",
                        "  levels: [0]",
                        "  kp: [1, 2]",
                        "  ki: [0, 1]",
                        "  kd: 0",
                        "  we: 1",
                        "  allow_steady_state_error: true",
                        "  method: pyomo",
                        "  generations: 1",
                        "  population: 4",
                        "",
                    ]
                )
            )
            with self.assertRaises(ValueError):
                load_config(root / "config.yaml")

    def test_default_config_resolves_from_the_repository_root(self):
        root = repository_root(Path(__file__).resolve().parents[1] / "app" / "identify.py")
        self.assertEqual(root, Path(__file__).resolve().parents[1])
        self.assertEqual(
            resolve_config(None, root),
            (root / "config" / "config.yaml").resolve(),
        )
        self.assertEqual(
            resolve_config("config/config.yaml", root),
            (root / "config" / "config.yaml").resolve(),
        )
