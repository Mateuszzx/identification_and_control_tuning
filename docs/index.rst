Plant identification and PID tuning
====================================

Fit a plant from a closed-loop proportional log, save it as JSON, then tune
the PID in :mod:`pid` on that plant. Identification does not choose gains,
and tuning does not fit a plant. Replace :meth:`pid.controller.PID.step` to
match your controller.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   quickstart
   usage

.. toctree::
   :maxdepth: 1
   :caption: API

   pid
   plantid
   pidtune
