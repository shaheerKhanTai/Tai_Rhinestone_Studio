# Building Rhinestone Studio for Windows

This guide explains how to build a standalone Windows executable from the source code.

## Prerequisites

- **Windows 10 or Windows 11**
- **Python 3.14 or later**
- **Git** (optional, for cloning the repository)

## Step-by-Step Build Instructions

### 1. Get the Source Code

If you have the source code folder, open PowerShell or Command Prompt in that directory.

If cloning from a repository:
```powershell
git clone <repository-url>
cd Tai_Rhinestone_Studio
```

### 2. Install Python Dependencies

Install all required packages including development tools:

```powershell
python -m pip install -e ".[dev]"
```

This installs:
- PySide6 (GUI framework)
- NumPy (numerical operations)
- OpenCV (image processing)
- Pillow (image I/O)
- ezdxf (DXF export)
- ReportLab (PDF generation)
- pytest (testing framework)
- PyInstaller (executable builder)

**Verification:** Run the test suite to ensure everything is installed correctly:

```powershell
python -m pytest
```

You should see: `401 passed` (may take 2-3 minutes)

### 3. Generate Application Icon

The Windows executable needs an `.ico` icon file:

```powershell
python -c "from build.make_icon import build_ico; build_ico()"
```

This creates: `app/resources/icons/app.ico`

**Verification:** Check that the file exists and is about 25 KB.

### 4. Build the Executable

Run PyInstaller with the provided specification file:

```powershell
pyinstaller rhinestone_studio.spec
```

**What happens:**
- PyInstaller analyzes the application
- Collects all dependencies
- Bundles Python runtime
- Creates the executable
- Packages everything into a folder

**Time:** 2-5 minutes depending on your system

**Output:** `dist/Rhinestone Studio/` folder containing:
- `Rhinestone Studio.exe` - The main executable
- `_internal/` - All dependencies and libraries
- Total size: ~100-300 MB

### 5. Test the Executable

Navigate to the output folder and run the application:

```powershell
cd "dist\Rhinestone Studio"
.\Rhinestone\ Studio.exe
```

The application should launch without any console window.

**Test checklist:**
- [ ] Application window opens
- [ ] Welcome screen displays correctly
- [ ] Import an image successfully
- [ ] Generate a simple design
- [ ] Export to SVG/DXF/PNG/PDF
- [ ] Save and load a project file
- [ ] Close and reopen the application

### 6. Distribute the Application

To share the application with users:

1. **Zip the entire folder:**
   ```powershell
   Compress-Archive -Path "dist\Rhinestone Studio" -DestinationPath "RhinestoneStudio-v0.1.0.zip"
   ```

2. **Or create an installer** (optional, requires additional tools like Inno Setup)

3. **Include documentation:**
   - Copy `README.md` to the distribution folder
   - Consider adding a `Quick Start Guide.pdf`

## Build Configuration

The build is controlled by `rhinestone_studio.spec`:

### Key Settings

- **Windowed mode:** `console=False` - No console window appears
- **One-folder mode:** Files are bundled in a folder, not a single executable
- **Icon:** `app/resources/icons/app.ico`
- **Excluded modules:** Tkinter, QtWebEngine (reduces size)
- **Hidden imports:** ezdxf, reportlab (lazy imports explicitly bundled)

### Customizing the Build

Edit `rhinestone_studio.spec` to modify:

**Application name:**
```python
name="Rhinestone Studio",  # Change the executable name
```

**Icon file:**
```python
icon=str(_icon) if _icon.exists() else None,
```

**Excluded modules** (to reduce size):
```python
excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtQuick"],
```

## Troubleshooting

### "Module not found" errors during build

**Solution:** Add missing modules to `hiddenimports` in the spec file:
```python
hiddenimports=[
    "ezdxf",
    "reportlab.pdfgen",
    "reportlab.graphics",
    "your_missing_module",  # Add here
],
```

### Build completes but executable crashes

**Diagnosis:** Run the executable from PowerShell to see error messages:
```powershell
cd "dist\Rhinestone Studio"
.\Rhinestone\ Studio.exe
```

**Common causes:**
- Missing data files → Add to `datas` in spec file
- Missing DLL → Install Visual C++ Redistributable
- Permission issues → Run as administrator

### "app.ico not found" warning

**Solution:** Regenerate the icon:
```powershell
python -c "from build.make_icon import build_ico; build_ico()"
```

### Executable is too large (>400 MB)

**Optimization:**
1. Add more modules to `excludes` in the spec file
2. Use `upx=True` for compression (already enabled)
3. Remove unused dependencies from `pyproject.toml`

### Application runs but UI looks wrong

**High-DPI displays:** The app is already configured for high-DPI support. If scaling is still wrong, check Windows display settings.

**Theme issues:** Try forcing light theme:
```powershell
.\Rhinestone\ Studio.exe --theme light
```

## Advanced: One-File Build

To create a single executable file instead of a folder:

1. Edit `rhinestone_studio.spec`, change:
   ```python
   exe = EXE(
       ...
       exclude_binaries=False,  # Changed from True
   )
   ```

2. Comment out or remove the `COLLECT` section

3. Rebuild: `pyinstaller rhinestone_studio.spec`

**Trade-offs:**
- **Pro:** Single file distribution
- **Con:** Slower startup (extracts to temp folder each time)
- **Con:** Larger file size (~300-400 MB)

## Build Automation

For CI/CD or automated builds, create a build script:

**`build.ps1` (PowerShell):**
```powershell
# Rhinestone Studio Build Script

Write-Host "Installing dependencies..."
python -m pip install -e ".[dev]"

Write-Host "Running tests..."
python -m pytest
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed! Aborting build."
    exit 1
}

Write-Host "Generating icon..."
python -c "from build.make_icon import build_ico; build_ico()"

Write-Host "Building executable..."
pyinstaller rhinestone_studio.spec

Write-Host "Creating distribution archive..."
Compress-Archive -Path "dist\Rhinestone Studio" -DestinationPath "RhinestoneStudio-v0.1.0.zip" -Force

Write-Host "Build complete! Output: RhinestoneStudio-v0.1.0.zip"
```

Run with: `.\build.ps1`

## Deployment Checklist

Before distributing to customers:

- [ ] All 401 tests pass
- [ ] Build completes without warnings
- [ ] Executable runs on a clean Windows machine (without Python installed)
- [ ] Test on Windows 10 and Windows 11
- [ ] Test on high-DPI and standard displays
- [ ] Verify all file operations (import, export, save, load)
- [ ] Check that settings persist between sessions
- [ ] Verify autosave and recovery work correctly
- [ ] Test with various image formats (PNG, JPG, BMP, TIFF, WebP)
- [ ] Confirm all export formats work (SVG, DXF, PNG, PDF)
- [ ] Include README.md and license information
- [ ] Create release notes documenting features

## File Structure After Build

```
dist/
└── Rhinestone Studio/
    ├── Rhinestone Studio.exe        # Main executable (~5 MB)
    ├── _internal/                   # All dependencies
    │   ├── PySide6/                 # Qt framework
    │   ├── cv2/                     # OpenCV
    │   ├── numpy/                   # NumPy
    │   ├── PIL/                     # Pillow
    │   ├── ezdxf/                   # DXF library
    │   ├── reportlab/               # PDF library
    │   └── ... (many other files)
    └── app/
        └── resources/               # Icons and styles
            └── icons/
                └── app.ico
```

**User data storage:** `%LOCALAPPDATA%\RhinestoneStudio\`
- `settings.json` - Application settings
- `autosave/` - Automatic project backups
- `logs/` - Application logs

## Support and Maintenance

### Updating Dependencies

To update all dependencies to their latest versions:

```powershell
python -m pip install --upgrade pip
python -m pip install --upgrade -e ".[dev]"
```

After updating, always:
1. Run the test suite: `python -m pytest`
2. Test the application manually
3. Rebuild the executable
4. Test the new executable on a clean machine

### Version Numbering

Update version in `app/__init__.py`:

```python
__version__ = "0.2.0"  # Update this
```

This version appears in:
- Window title
- About dialog
- Project files
- Logs

### Building for Other Platforms

**Note:** This spec file is Windows-specific. For macOS or Linux:

1. Create a new spec file for that platform
2. Adjust paths (use `pathlib` for cross-platform compatibility)
3. Change icon format (`.icns` for macOS, `.png` for Linux)
4. Test thoroughly on the target platform

---

**Questions?** Review the PyInstaller documentation: https://pyinstaller.org/

**Build issues?** Check the `build/` folder for helper scripts and examples.
