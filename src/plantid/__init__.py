"""Plant identification from a closed-loop log.

PID search lives in ``pidtune``. The controller lives in ``pid``. This
package fits the plant and writes it as JSON. It does not choose the gains.

:class:`~plantid.plant.Plant` is the fitted dynamics. :class:`~plantid.log.Log`
reads the recording and :meth:`~plantid.log.Log.identify` fits the structures
from :meth:`~plantid.models.Model.read`.
"""

from plantid.config import Config, IdentifySetup, TuneSetup, load_config
from plantid.fit import Fit, FitResult
from plantid.log import Log
from plantid.models import Model
from plantid.plant import Plant

__all__ = [
    "Config",
    "Fit",
    "FitResult",
    "IdentifySetup",
    "Log",
    "Model",
    "Plant",
    "TuneSetup",
    "load_config",
]
