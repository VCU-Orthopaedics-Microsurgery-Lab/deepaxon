"""
utils/__main__.py

DeepAxon utility library. Most modules are imported internally by the pipeline.
Two utilities are available as standalone command-line tools:

    python -m utils.version
    python -m utils.class_balance --masks path/to/masks/

Modules:
    helpers        Shared functions (config, scan_study, batch sizing, Hann step)
    version        Version string and environment fingerprint (CLI tool)
    gpu            GPU detection and device panel
    logger         Unified Rich console and file logging
    resize         Image resize with interpolation policy (LANCZOS4 / NEAREST)
    class_balance  Pixel class balance report across BGW masks (CLI tool)
"""

print(__doc__)