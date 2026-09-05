# Rhinestone Studio

**Professional Image-to-Rhinestone Design Software**

Convert raster images into editable rhinestone designs and export production-ready files for shirts, costumes, tumblers, caps, craft projects and Cricut / Silhouette workflows.

Rhinestone Studio is an **offline-first Windows desktop application**. Your images never leave your computer — there is no telemetry and no network access.

> **Version:** 0.1.0  
> **Status:** Fully functional commercial-ready application with 401 passing tests

---

## Features

### Core Capabilities
- **Import images** - PNG, JPG, BMP, TIFF, WebP, and SVG (vector) support
- **Set physical dimensions** - Work in inches, millimeters, or centimeters
- **Configure rhinestones** - SS4 through SS30 standard sizes, plus custom diameters
- **Multiple placement modes** - Grid, Offset/Brick, Hexagonal, and Organic algorithms
- **Manual editing** - Add, move, delete stones with full undo/redo support
- **Collision detection** - Automatic detection and resolution of overlapping stones
- **Production exports** - SVG, Cricut SVG, DXF, PNG preview, and PDF production sheets

### Design Controls
- **Detail control** - Adjust how much image detail is captured
- **Density control** - Control stone distribution density
- **Contrast adjustment** - Enhance or reduce contrast in the design
- **Smoothing** - Reduce noise and create cleaner designs
- **Edge preservation** - Maintain sharp edges in your design
- **Maximum stone count** - Limit designs to a specific stone count
- **Invert mode** - For light subjects on dark backgrounds

### Professional Outputs
- **SVG** - Real vector circles at true physical dimensions
- **Cricut SVG** - Optimized for Cricut Design Space with hole allowance
- **DXF** - AutoCAD-compatible vector format for laser cutters
- **PNG** - High-resolution preview (300 DPI)
- **PDF** - Professional production sheet with design specifications

---

## Workflow

```
Import Image → Crop / Transform → Set Physical Size → Configure Rhinestones
   → Generate → Preview → Edit → Validate → Export
```

The application guides you through this workflow with a clear left-to-right interface:
1. **Image** - Import and adjust your source image
2. **Design Size** - Set physical dimensions and units
3. **Rhinestones** - Choose stone size, spacing, and placement mode
4. **Appearance** - Fine-tune density, detail, and visual properties
5. **Generate** - Create your rhinestone design
6. **Edit** - Manually refine the design
7. **Export** - Save production-ready files

---

## Installation

### For Users (Standalone Application)

1. Download the latest release from the Releases page
2. Extract the `Rhinestone Studio` folder
3. Run `Rhinestone Studio.exe`
4. No Python installation required!

### For Developers

**Requirements:**
- Python 3.14+
- Windows 10/11

**Setup:**

```powershell
# Clone or download the repository
cd Tai_Rhinestone_Studio

# Install dependencies
python -m pip install -e ".[dev]"

# Run the test suite (401 tests)
python -m pytest

# Run the application
python -m app.main
```

---

## Building from Source

To create a standalone Windows executable:

```powershell
# 1. Install dependencies (includes PyInstaller)
python -m pip install -e ".[dev]"

# 2. Generate the application icon
python -c "from build.make_icon import build_ico; build_ico()"

# 3. Build the executable
pyinstaller rhinestone_studio.spec

# 4. Find your application in:
# dist/Rhinestone Studio/Rhinestone Studio.exe
```

The build creates a one-folder, windowed (no console) application. All settings, logs, and autosave files are stored in `%LOCALAPPDATA%\RhinestoneStudio`, so the installation folder can be read-only.

---

## Quick Start Guide

### 1. Import an Image

Click **Import Image** or use `File → Import Image`. Supported formats:
- PNG (recommended for logos and designs)
- JPG/JPEG
- BMP
- TIFF
- WebP
- SVG (vector — rasterised automatically at import, aspect ratio and transparency preserved)

**Tips:**
- High-contrast images work best
- PNG with transparency is fully supported
- SVG files keep their `viewBox` aspect ratio and never stretch; transparent areas of an SVG stay transparent
- Ideal resolution: 500-2000px for the longest dimension

### 2. Adjust the Image (Optional)

Use the adjustment tools in the left panel:
- **Rotate** - 90° clockwise/counterclockwise
- **Flip** - Horizontal or vertical
- **Crop** - Use the crop tool in the toolbar to select your design area
- **Reset** - Return to original

### 3. Set Physical Size

Choose a preset or enter custom dimensions:
- **Presets:** T-Shirt Front, T-Shirt Pocket, Cap, Tumbler, etc.
- **Units:** inches, mm, or cm
- **Lock aspect ratio** - Maintain proportions when resizing

**Common sizes:**
- T-Shirt Front: 10" × 12"
- T-Shirt Pocket: 4" × 4"
- Cap: 3.5" × 2"
- Tumbler: 8.5" × 3"

### 4. Configure Rhinestones

**Stone Size:**
- SS6 (2.0mm) - Very small, high detail
- SS10 (2.8mm) - Most popular size
- SS16 (4.0mm) - Medium, good visibility
- SS20 (5.0mm) - Large, bold designs
- Custom - Specify exact diameter

**Spacing:** Gap between stones (typically 0.5mm)

**Placement Mode:**
- **Grid** - Regular rows and columns
- **Offset** - Brick pattern (every other row offset)
- **Hexagonal** - Most efficient packing (recommended)
- **Organic** - Follows image density naturally

### 5. Fine-Tune Appearance

Adjust sliders to control the design:

- **Detail** (0-1) - Higher = captures more fine details
- **Density** (0-1) - Higher = more stones placed
- **Contrast** (0-1) - Higher = stronger light/dark differences
- **Smoothing** (0-1) - Higher = smoother, less noisy
- **Edge Preservation** (0-1) - Higher = sharper edges

**Optional:** Enable **Limit stones** to cap the total count (e.g., 5000 stones max)

### 6. Generate Design

Click **Generate Design** - this takes 5-30 seconds depending on complexity.

Watch the progress bar for generation stages:
- Analyzing image
- Generating density map
- Placing stones
- Checking collisions
- Optimizing
- Finalizing

### 7. Preview and Edit

The canvas shows your design with individual rhinestones as circles.

**Canvas controls:**
- **Mouse wheel** - Zoom in/out
- **Space + drag** - Pan
- **Fit to screen** - View entire design

**Editing tools:**
- **Select** - Click to select stones
- **Move** - Drag selected stones
- **Add stone** - Double-click or use Add Stone tool
- **Delete** - Select stones and press Delete
- **Brush Add** - Paint stones with your cursor
- **Eraser** - Remove stones by brushing

**Background overlay:**
- Toggle **Show Original Image** to compare with source
- Adjust opacity to see design over the original

### 8. Export

Choose your export format:

**File → Export:**
- **SVG** - General vector format (Illustrator, Inkscape, laser cutters)
- **Cricut SVG** - Optimized for Cricut Design Space
- **DXF** - AutoCAD format (laser cutters, CNC)
- **PNG Preview** - High-resolution image (300 DPI)
- **PDF Production Sheet** - Professional spec sheet with design info
- **Production Package** - All formats in one folder

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Import Image | `Ctrl+O` |
| Save Project | `Ctrl+S` |
| Export | `Ctrl+E` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Select All | `Ctrl+A` |
| Delete Selected | `Delete` |
| Zoom In | `Ctrl++` or `Mouse Wheel Up` |
| Zoom Out | `Ctrl+-` or `Mouse Wheel Down` |
| Fit to Screen | `Ctrl+0` |
| Pan Canvas | `Space + Drag` |

---

## Project Files

Rhinestone Studio saves projects with the `.rhinestone` extension. These files preserve:
- Source image reference
- All transformations (crop, rotate, flip)
- Physical dimensions and units
- Rhinestone settings
- Generated stone positions
- Complete design state for reopening

**Autosave:** Projects are automatically saved every 5 minutes to prevent data loss.

**Recovery:** If the application closes unexpectedly, you'll be offered to recover your work on restart.

---

## Tips for Best Results

1. **Start with high-contrast images** - Clear subject on plain background works best
2. **Use PNG with transparency** - Automatically excludes background
3. **Test with lower density first** - Generate faster, then increase if needed
4. **Hexagonal placement** - Generally produces the best-looking, most efficient designs
5. **Limit stone count** - For beginners, start with 2000-3000 stones
6. **Preview on canvas** - Always review the design before exporting
7. **Manual editing** - Don't hesitate to add/remove individual stones for perfection
8. **Save often** - Use `Ctrl+S` to save your project regularly

---

## Architecture

Layered architecture with strict separation of concerns:

```
app/
├── core/       Image processing, placement, collision, geometry (no Qt)
├── models/     Typed domain objects: Project, Design, Stone, Settings
├── exporters/  SVG / DXF / PNG / PDF writers
├── services/   Project I/O, autosave, export orchestration, settings
├── ui/         PySide6 interface (main window, canvas, panels, dialogs)
└── resources/  Icons and QSS styles

tests/          pytest suite (401 tests)
```

**Internal unit:** Millimeters for all physical calculations. The UI converts to/from the user's chosen display unit (inches / cm / mm).

**Design principle:** The GUI contains no generation algorithms. The rhinestone engine is independently testable without starting Qt.

---

## Support

For issues, feature requests, or questions:
- Check the built-in Help menu
- Review this README
- Examine the test suite for usage examples

---

## Technical Details

- **Language:** Python 3.14+
- **GUI Framework:** PySide6 (Qt 6)
- **Image Processing:** OpenCV, NumPy, Pillow
- **Vector Export:** ezdxf (DXF), custom SVG generation
- **PDF Generation:** ReportLab
- **Testing:** pytest (401 tests, 100% passing)
- **Build System:** PyInstaller for Windows executables

---

## License

Proprietary. All rights reserved.

This is a commercial software product intended for sale on Etsy and similar marketplaces.

---

## Version History

### Version 0.1.0 (Current)
- Initial release
- Full image-to-rhinestone conversion pipeline
- Four placement modes (Grid, Offset, Hexagonal, Organic)
- Complete manual editing with undo/redo
- All export formats (SVG, Cricut SVG, DXF, PNG, PDF)
- Project save/load with autosave
- 401 comprehensive tests
- Professional UI with dark theme
- Collision detection and resolution
- Windows packaging ready

---

**Rhinestone Studio** — Create professional rhinestone designs in minutes.
