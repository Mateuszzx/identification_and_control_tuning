import tempfile
import unittest
from pathlib import Path

import numpy as np

from plantid.plots import save_identify, save_tune

PNG = b"\x89PNG\r\n\x1a\n"


class PlotTest(unittest.TestCase):
    def test_identify_figure_is_a_png(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "identify.png"
            save_identify(path, [0.0, 1.0], [0.0, 1.0], [0.0, 0.4], [0.0, 0.5], "model")
            self.assertTrue(path.is_file())
            self.assertTrue(path.read_bytes().startswith(PNG))

    def test_tune_figure_is_a_png(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tune.png"
            time = np.array([0.0, 0.015])
            save_tune(path, time, [0.0, 1.0], [0.0, 0.2], [0.0, 10.0])
            self.assertTrue(path.is_file())
            self.assertTrue(path.read_bytes().startswith(PNG))
