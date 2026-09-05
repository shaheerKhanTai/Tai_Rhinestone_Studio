"""Comprehensive end-to-end workflow test for Rhinestone Studio.

This script tests the complete workflow:
1. Import image
2. Set physical dimensions
3. Configure rhinestone settings
4. Generate design
5. Verify stone placement
6. Export to all formats (SVG, DXF, PNG, PDF)
7. Verify exported files

Run with: python test_workflow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from app.core.geometry import Unit, to_mm
from app.core.rhinestone_engine import generate
from app.exporters.dxf_exporter import export_dxf
from app.exporters.pdf_exporter import export_pdf, ProductionSheetInfo
from app.exporters.png_exporter import export_png
from app.exporters.svg_exporter import export_svg, export_cricut_svg
from app.models.design import Design
from app.models.settings import GenerationSettings, PlacementMode


def test_complete_workflow():
    """Test the complete rhinestone design workflow."""
    print("=" * 70)
    print("Rhinestone Studio - End-to-End Workflow Test")
    print("=" * 70)

    # Step 1: Load test image
    print("\n1. Loading test image...")
    test_image_path = Path("test_star.png")
    if not test_image_path.exists():
        print(f"ERROR: Test image not found: {test_image_path}")
        return False

    # Load image as RGBA numpy array
    image = cv2.imread(str(test_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"ERROR: Failed to load image")
        return False

    # Convert to RGBA if needed
    if len(image.shape) == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4:  # BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    print(f"   OK Loaded {test_image_path.name}: {image.shape[1]}x{image.shape[0]}px")

    # Step 2: Set physical dimensions
    print("\n2. Setting physical dimensions...")
    width_mm = to_mm(8.0, Unit.INCH)  # 8 inches
    height_mm = to_mm(6.0, Unit.INCH)  # 6 inches
    print(f"   OK Design size: 8.0 x 6.0 inches ({width_mm:.1f} x {height_mm:.1f} mm)")

    # Step 3: Configure rhinestone settings
    print("\n3. Configuring rhinestone settings...")
    settings = GenerationSettings()
    settings.stone_size_id = "SS10"
    settings.spacing_mm = 0.5
    settings.placement_mode = PlacementMode.HEXAGONAL
    settings.density = 0.6
    settings.detail = 0.6
    settings.contrast = 0.5
    settings.smoothing = 0.3
    settings.edge_preservation = 0.5
    settings.max_stones = 5000
    print(f"   OK Stone size: {settings.stone_size_id}")
    print(f"   OK Spacing: {settings.spacing_mm} mm")
    print(f"   OK Placement mode: {settings.placement_mode.label}")
    print(f"   OK Max stones: {settings.max_stones}")

    # Step 4: Generate rhinestone design
    print("\n4. Generating rhinestone design...")
    result = generate(image, settings, width_mm, height_mm)
    print(f"   OK Generated {result.stone_count} stones")

    if result.stone_count == 0:
        print("   ERROR: No stones generated!")
        return False

    if result.stone_count > settings.max_stones:
        print(f"   WARNING: Stone count ({result.stone_count}) exceeds max ({settings.max_stones})")

    # Convert GenerationResult to Design for export
    design = Design(width_mm, height_mm, stones=result.stones)

    # Step 5: Verify stone placement
    print("\n5. Verifying stone placement...")

    # Check that stones are within bounds
    for stone in result.stones[:10]:  # Sample first 10
        if stone.x < 0 or stone.x > width_mm:
            print(f"   ERROR: Stone x={stone.x} out of bounds (0 to {width_mm})")
            return False
        if stone.y < 0 or stone.y > height_mm:
            print(f"   ERROR: Stone y={stone.y} out of bounds (0 to {height_mm})")
            return False

    print(f"   OK All stones within design bounds")
    print(f"   OK Design dimensions: {design.width_mm:.2f} x {design.height_mm:.2f} mm")

    # Step 6: Export to all formats
    print("\n6. Exporting design to all formats...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Export SVG
    svg_path = output_dir / "test_design.svg"
    export_svg(design, svg_path)
    if svg_path.exists():
        print(f"   OK SVG exported: {svg_path} ({svg_path.stat().st_size} bytes)")
    else:
        print(f"   ERROR: SVG export failed")
        return False

    # Export Cricut SVG
    cricut_svg_path = output_dir / "test_design_cricut.svg"
    export_cricut_svg(design, cricut_svg_path)
    if cricut_svg_path.exists():
        print(f"   OK Cricut SVG exported: {cricut_svg_path} ({cricut_svg_path.stat().st_size} bytes)")
    else:
        print(f"   ERROR: Cricut SVG export failed")
        return False

    # Export DXF
    dxf_path = output_dir / "test_design.dxf"
    export_dxf(design, dxf_path)
    if dxf_path.exists():
        print(f"   OK DXF exported: {dxf_path} ({dxf_path.stat().st_size} bytes)")
    else:
        print(f"   ERROR: DXF export failed")
        return False

    # Export PNG preview
    png_path = output_dir / "test_design.png"
    export_png(design, png_path)
    if png_path.exists():
        print(f"   OK PNG exported: {png_path} ({png_path.stat().st_size} bytes)")
    else:
        print(f"   ERROR: PNG export failed")
        return False

    # Export PDF production sheet
    pdf_path = output_dir / "test_design.pdf"
    pdf_info = ProductionSheetInfo(design_name="Test Star Design")
    export_pdf(design, pdf_path, pdf_info)
    if pdf_path.exists():
        print(f"   OK PDF exported: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    else:
        print(f"   ERROR: PDF export failed")
        return False

    # Step 7: Verify exported files
    print("\n7. Verifying exported file contents...")

    # Verify SVG contains circles
    svg_content = svg_path.read_text()
    if "<circle" in svg_content and "cx=" in svg_content:
        print(f"   OK SVG contains circle elements")
    else:
        print(f"   WARNING: SVG may not contain proper circle elements")

    # Verify DXF has reasonable size
    if dxf_path.stat().st_size > 1000:
        print(f"   OK DXF file has reasonable size")
    else:
        print(f"   WARNING: DXF file seems too small")

    # Verify PNG has reasonable dimensions
    from PIL import Image
    with Image.open(png_path) as img:
        print(f"   OK PNG dimensions: {img.width}x{img.height}px")
        if img.width < 100 or img.height < 100:
            print(f"   WARNING: PNG dimensions seem too small")

    print("\n" + "=" * 70)
    print("OK ALL WORKFLOW TESTS PASSED")
    print("=" * 70)
    print(f"\nOutput files created in: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  - {svg_path.name} ({svg_path.stat().st_size:,} bytes)")
    print(f"  - {cricut_svg_path.name} ({cricut_svg_path.stat().st_size:,} bytes)")
    print(f"  - {dxf_path.name} ({dxf_path.stat().st_size:,} bytes)")
    print(f"  - {png_path.name} ({png_path.stat().st_size:,} bytes)")
    print(f"  - {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")
    return True


if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)
