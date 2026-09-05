"""Production exporters: SVG, Cricut SVG, PNG preview, DXF, PDF.

Each format is an independent module so its rules stay isolated. Exporters
consume the physical geometry of a :class:`~app.models.design.Design` (stone
positions and diameters in millimetres) and emit documents in the correct
physical dimensions — never a screenshot.
"""
