"""Application services: project I/O, autosave, export orchestration, settings.

Services coordinate the core engine, models and exporters on behalf of the UI,
and own side effects like the filesystem and logging. They must not import from
:mod:`app.ui`.
"""
