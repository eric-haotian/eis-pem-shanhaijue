# learning AI at www.haotianblog.com
"""Vendored physics layer for the ShanHaiJue / ShanHaiJian distribution.

A single flat namespace holding the 48-parameter frequency-domain impedance
model, its parameter/prior machinery, and the numerical utilities they need.
Upstream these modules lived in two packages (a physics foundation and a layer
built on top of it); they are merged here because a distribution has no use for
that split, and the package initialiser is deliberately empty so that importing
one module does not drag in unrelated ones.

Import modules directly, e.g.::

    from eispem.seis_model import StackedSEISModel
    from eispem.seis_pipeline import SEISPhysicsPriorBuilder
"""
