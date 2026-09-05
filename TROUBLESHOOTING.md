# Troubleshooting Guide - Rhinestone Studio

This guide helps resolve common issues users may encounter.

## Installation & Launch Issues

### Application won't start / shows error on launch

**Symptoms:** Double-clicking the executable does nothing, or shows an error message.

**Solutions:**

1. **Check Windows version**
   - Requires Windows 10 or Windows 11
   - Right-click Start → System to check your version

2. **Install Visual C++ Redistributable**
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run the installer
   - Restart your computer
   - Try launching Rhinestone Studio again

3. **Check antivirus/Windows Defender**
   - Some antivirus software blocks unrecognized executables
   - Add Rhinestone Studio to your antivirus exceptions
   - Or temporarily disable antivirus to test

4. **Run as Administrator**
   - Right-click `Rhinestone Studio.exe`
   - Select "Run as administrator"
   - If this works, check folder permissions

5. **Check for corrupted download**
   - Re-download the application
   - Verify file size matches the original
   - Extract to a different location

### "Application is not responding" on startup

**Cause:** Windows Defender or antivirus scanning the application.

**Solution:** Wait 30-60 seconds on first launch. Subsequent launches will be faster.

---

## Image Import Issues

### "Failed to load image" error

**Causes:**
- Corrupted image file
- Unsupported format
- File locked by another program

**Solutions:**

1. **Verify file format**
   - Supported: PNG, JPG, JPEG, BMP, TIFF, WebP, SVG
   - Check file extension matches actual format
   - For SVG: ensure the file is valid XML with an `<svg>` root element

2. **Test image in another program**
   - Open image in Windows Photos or Paint
   - If it doesn't open, the file is corrupted

3. **Convert image format**
   - Save as PNG using Paint or another image editor
   - PNG is the most reliable format

4. **Check file permissions**
   - Ensure you can read the file
   - Try copying to your Desktop first

### Image appears black or blank after import

**Cause:** Fully transparent image or extremely dark image.

**Solutions:**

1. **Check original image**
   - Open in an image viewer
   - Verify it has visible content

2. **Adjust contrast slider**
   - In the Appearance section, increase Contrast to 0.7-0.9
   - Try inverting if you have a light subject on dark background

3. **Check image mode**
   - If using a transparent PNG, ensure there's visible content in the non-transparent areas

---

## Generation Issues

### "No stones generated" or very few stones

**Causes:**
- Image is too light/dark
- Density set too low
- Max stones limit too restrictive

**Solutions:**

1. **Increase Density**
   - Slide Density to 0.7-0.9
   - Click Generate Design again

2. **Adjust Detail**
   - Increase Detail slider to 0.6-0.8
   - This captures more of the image

3. **Check Contrast**
   - Increase Contrast to make subject more prominent
   - Try values between 0.5-0.8

4. **Remove or increase Max Stones limit**
   - Uncheck "Limit stones" checkbox
   - Or increase the limit to 10,000+

5. **Check image content**
   - Ensure image has sufficient contrast
   - High-contrast images work best

### Generation is very slow (more than 1 minute)

**Causes:**
- Very large design dimensions
- High stone count
- Complex image with high detail

**Solutions:**

1. **Reduce design size**
   - Try 8" × 8" instead of 20" × 20"
   - Smaller designs process faster

2. **Increase stone size**
   - Use SS16 or SS20 instead of SS6 or SS10
   - Larger stones = fewer stones = faster generation

3. **Reduce detail**
   - Lower Detail slider to 0.3-0.5
   - This simplifies the processing

4. **Enable Max Stones limit**
   - Check "Limit stones"
   - Set to 3000-5000 stones
   - This caps generation time

5. **Close other programs**
   - Free up system memory
   - Close web browsers and other applications

### Generation crashes the application

**Cause:** Insufficient memory for very large designs.

**Solutions:**

1. **Reduce design size** to 10" × 10" or smaller
2. **Enable Max Stones** limit to 5000
3. **Close other programs** to free memory
4. **Restart the application** and try again
5. **Use a smaller source image** (resize to 1000px before importing)

---

## Export Issues

### "Export failed" error

**Causes:**
- No design generated yet
- Disk full or write-protected
- File in use by another program

**Solutions:**

1. **Generate design first**
   - Click "Generate Design" before exporting
   - Wait for generation to complete

2. **Check disk space**
   - Ensure you have at least 100 MB free
   - Export to a different drive if needed

3. **Change export location**
   - Don't export to protected folders (C:\Program Files)
   - Use Desktop or Documents folder

4. **Close other programs**
   - If exporting DXF, close AutoCAD or other CAD software
   - If exporting PDF, close Adobe Reader

### Exported SVG doesn't open correctly in Cricut Design Space

**Issue:** Cricut can't read the standard SVG format.

**Solution:** Use **File → Export → Cricut SVG** instead of regular SVG.

**The Cricut SVG:**
- Uses inches instead of millimeters
- Includes hole allowance for stencils
- Is optimized for Cricut's requirements

### Exported DXF has wrong dimensions

**Cause:** CAD software interpreting units incorrectly.

**Solutions:**

1. **Check DXF units**
   - Open in CAD software
   - Verify units are set to millimeters
   - The file is exported in millimeters

2. **Measure a known dimension**
   - If you designed an 8" × 6" design, it should be 203.2mm × 152.4mm
   - If dimensions are wrong, scale appropriately

3. **Use scale factor**
   - If dimensions are in inches when they should be mm: scale by 25.4
   - If dimensions are in mm when they should be inches: scale by 0.0394

### PDF production sheet is blank or incomplete

**Cause:** Font issues or PDF rendering problem.

**Solutions:**

1. **Update to latest version** of Rhinestone Studio
2. **Try exporting PNG** instead for visual preview
3. **Check PDF in different viewer** (Adobe Reader, Edge, Chrome)

---

## Editing Issues

### Can't select or move stones

**Cause:** Wrong tool selected or stones are locked.

**Solutions:**

1. **Check active tool**
   - Click the Select tool (arrow icon) in toolbar
   - Should be highlighted when active

2. **Zoom in**
   - Stones may be too small to click at current zoom
   - Use mouse wheel or Ctrl+= to zoom in

3. **Click directly on stone center**
   - Click the center of the circle, not the edge
   - Try rectangular selection (drag to select multiple)

### Undo (Ctrl+Z) doesn't work

**Cause:** Action isn't undoable or history limit reached.

**Solutions:**

1. **Verify you're undoing a design change**
   - Undo works for: stone add/delete/move, generation
   - Undo doesn't work for: window resize, zoom, pan

2. **Save your project frequently**
   - Use Ctrl+S to save
   - You can reload if you make a mistake

3. **Restart if undo is broken**
   - Close and reopen the application
   - Load your last saved project

### Adding stones manually is tedious

**Solution:** Use the **Brush Add tool**
- Click the Brush Add tool in toolbar
- Click and drag to "paint" stones
- Much faster than clicking individual Add Stone positions

---

## Performance Issues

### Application is slow or laggy with many stones

**Causes:**
- Very high stone count (10,000+)
- Low-end graphics hardware
- Other applications using system resources

**Solutions:**

1. **Reduce stone count**
   - Enable "Limit stones" to 5000
   - Use larger stone sizes (SS16 or SS20)

2. **Turn off background image**
   - Uncheck "Show Original Image"
   - This reduces rendering load

3. **Zoom to specific area**
   - Don't try to view entire design at once
   - Zoom in to work on small sections

4. **Close other programs**
   - Free up RAM and CPU
   - Close web browsers, video players, etc.

5. **Update graphics drivers**
   - Visit your graphics card manufacturer's website
   - Download latest drivers

### Canvas zoom or pan is jerky

**Cause:** High stone count or graphics performance.

**Solutions:**

1. **Use keyboard shortcuts**
   - Ctrl+0 to fit to screen
   - Faster than zooming with wheel

2. **Reduce stone count** as described above

3. **Disable animations** (if option exists in Settings)

---

## Project File Issues

### "Failed to open project" error

**Causes:**
- Corrupted project file
- File created with newer version
- File permissions issue

**Solutions:**

1. **Check file extension**
   - Should be `.rhinestone`
   - If wrong, rename to proper extension

2. **Try opening autosave**
   - Look in `%LOCALAPPDATA%\RhinestoneStudio\autosave\`
   - Open the most recent autosave file

3. **Check version compatibility**
   - Ensure you have the latest version of Rhinestone Studio
   - Files from newer versions can't open in older versions

### Source image missing when opening project

**Issue:** Project opens but shows "Image file not found"

**Cause:** Original image was moved or deleted after creating project.

**Solutions:**

1. **Locate original image**
   - Check where you originally saved it
   - Look in Downloads, Desktop, Documents

2. **Re-import the image**
   - Click "Import Image"
   - Select the original image file
   - The design (stones) is preserved

3. **Use embedded image** (if feature exists)
   - Future versions may embed the image in the project file

### Autosave isn't working

**Check:**
1. Open **File → Settings**
2. Verify **Autosave** is enabled
3. Check **Autosave interval** (default: 5 minutes)
4. Make at least one change before autosave triggers

**Find autosaved projects:**
- `%LOCALAPPDATA%\RhinestoneStudio\autosave\`
- Files are named with timestamp

---

## Display Issues

### UI text is too small on high-DPI display

**Cause:** Windows display scaling not applied correctly.

**Solutions:**

1. **Check Windows scaling**
   - Right-click Desktop → Display settings
   - Verify Scaling is set correctly (usually 125% or 150%)

2. **Override high-DPI scaling** (if issue persists)
   - Right-click `Rhinestone Studio.exe`
   - Properties → Compatibility
   - Change high DPI settings
   - Check "Override high DPI scaling behavior"
   - Select "System (Enhanced)"

### Application appears blurry

**Cause:** Windows scaling issue.

**Solution:**
- Right-click `Rhinestone Studio.exe`
- Properties → Compatibility
- Uncheck "Disable fullscreen optimizations"
- Check "Override high DPI scaling behavior"
- Select "Application"

### Dark theme is too dark / light theme is too light

**Solution:** Switch themes in **View → Theme** menu
- Options: Light, Dark
- Or use command line: `Rhinestone Studio.exe --theme light`

---

## Error Messages

### "Dimensions must be positive"

**Cause:** Design width or height set to 0 or negative value.

**Solution:** Enter valid dimensions (e.g., 10.00 inches, 254 mm)

### "Invalid stone size"

**Cause:** Custom stone diameter set to invalid value.

**Solution:** 
- Set custom diameter between 0.5mm and 20mm
- Or choose a standard size (SS6 - SS30)

### "Collision detected"

**Info:** This is a warning, not an error. It means stones are overlapping.

**Solutions:**
- Click "Resolve Automatically" button
- Manually move stones apart
- Increase spacing in Rhinestones section

### "Failed to write file"

**Causes:**
- Insufficient disk space
- File or folder is read-only
- No permission to write to that location

**Solutions:**
- Check disk space (need at least 100 MB)
- Choose a different save location (Desktop, Documents)
- Run application as Administrator (if saving to protected location)

---

## Getting Help

If your issue isn't listed here:

1. **Check the main README.md** for general usage information
2. **Review the Quick Start Guide** in the application
3. **Check application logs:**
   - Location: `%LOCALAPPDATA%\RhinestoneStudio\logs\`
   - Look for ERROR or CRITICAL entries
   - Include these in support requests

4. **Restart the application**
   - Many transient issues are resolved by restarting
   - Make sure to save your work first

5. **Reinstall the application**
   - Your projects and settings are stored separately
   - Reinstalling won't delete your work
   - Just extract the new version over the old one

---

## Known Limitations

These are design limitations, not bugs:

1. **No multi-color designs** (yet)
   - All stones are single color
   - Export and manually color in vector software

2. **No batch processing**
   - Process one design at a time
   - Save project for each design

3. **No cloud sync**
   - Projects are local only (by design for privacy)
   - Manually back up important projects

4. **No printing support**
   - Export to PDF or PNG for printing
   - Use printer settings to set scale

5. **Windows only**
   - Currently Windows 10/11 only
   - macOS and Linux support may come in future versions

---

## Tips for Avoiding Issues

1. **Save frequently** - Use Ctrl+S regularly
2. **Start simple** - Test with small designs first
3. **Use presets** - Built-in size presets are reliable
4. **Keep backups** - Copy important projects to another location
5. **Close other apps** - Free up system resources when working with large designs
6. **Use PNG format** - Most reliable image format
7. **Test exports** - Always verify exported files open correctly
8. **Read tooltips** - Hover over controls for helpful information

---

**Still having issues?** Include these details when seeking help:
- Rhinestone Studio version
- Windows version
- Description of what you were doing
- Full error message (if any)
- Screenshot of the issue (if applicable)
- Application log file (from `%LOCALAPPDATA%\RhinestoneStudio\logs\`)
