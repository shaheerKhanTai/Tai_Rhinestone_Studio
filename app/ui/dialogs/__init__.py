"""Modal dialogs for Rhinestone Studio (settings, and future export/crop dialogs).

Dialogs are thin *views* over the models: they read a value object in, let the
user edit it, and hand an updated value object back to the caller. They contain
no persistence or generation logic themselves.
"""
