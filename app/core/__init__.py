"""Core engine: image processing, placement, collision, optimisation, geometry.

Everything in this package is independent of PySide6 and can be exercised
without starting the GUI. This is a hard architectural rule — do not import
from :mod:`app.ui` here.
"""
