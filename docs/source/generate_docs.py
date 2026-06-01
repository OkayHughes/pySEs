#!/usr/bin/env python3
"""Generate API documentation automatically."""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sphinx.ext.autosummary.generate import generate_autosummary_docs

# List of modules to document
modules = [
    'pyses',
    'pyses.analytic_initialization',
    'pyses.dynamical_cores',
    'pyses.dynamical_cores.cam_se',
    'pyses.dynamical_cores.homme',
    'pyses.dynamical_cores.tracer_advection',
    'pyses.mesh_generation',
    'pyses.mpi',
    'pyses.operations_2d',
    # ``pyses.shallow_water_models`` is deprecated -- subsumed by
    # ``models.shallow_water`` in the 3-D dynamical core.  The package
    # is kept as a thin compat shim during the deprecation window; once
    # all callers have migrated it (and these auto-generated stubs)
    # will be removed.
    'pyses.tracer_transport',
]

# Output directory for generated rst files
output_dir = 'api'

for module in modules:
    generate_autosummary_docs(
        [module],
        output_dir=output_dir,
        suffix='.rst',
        base_path='.'
    )

print("API documentation generated successfully!")