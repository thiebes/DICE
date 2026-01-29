# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DICE GUI application.

Build command:
    pyinstaller dice-gui.spec

Output:
    dist/DICE.exe (Windows)
    dist/DICE (Linux)
    dist/DICE.app (macOS)
"""

import sys
import os
from pathlib import Path

# Configuration
onefile = True
app_name = 'DICE'

block_cipher = None
root_path = Path(SPECPATH)

# Platform-specific settings
if sys.platform == 'win32':
    icon_file = str(root_path / 'webapp' / 'static' / 'favicon.ico')
    exe_name = f'{app_name}.exe'
elif sys.platform == 'darwin':
    icon_file = None  # Would need .icns file for macOS
    exe_name = app_name
else:
    icon_file = str(root_path / 'webapp' / 'static' / 'favicon.ico')
    exe_name = app_name

# Collect missing DLLs from conda environment (Windows)
binaries = []
if sys.platform == 'win32':
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    conda_dlls = Path(conda_prefix) / 'Library' / 'bin'
    if conda_dlls.exists():
        # DLLs that PyInstaller may miss from conda
        missing_dlls = [
            'libexpat.dll',
            'liblzma.dll',
            'LIBBZ2.dll',
            'ffi.dll',
            'sqlite3.dll',
            'zlib.dll',
        ]
        for dll in missing_dlls:
            dll_path = conda_dlls / dll
            if dll_path.exists():
                binaries.append((str(dll_path), '.'))

# Data files to include
datas = [
    (str(root_path / 'logo'), 'logo'),
    (str(root_path / 'graphics'), 'graphics'),
    (str(root_path / 'data'), 'data'),
]

# Hidden imports that PyInstaller may miss
hiddenimports = [
    # DICE core packages
    'dice',
    'dice.core',
    'dice.core.diffusion',
    'dice.core.noise',
    'dice.core.profiles',
    'dice.core.fitting',
    'dice.analysis',
    'dice.analysis.simulation',
    'dice.analysis.statistics',
    'dice.io',
    'dice.io.parameters',
    'dice.io.results',
    'dice.io.data_loader',
    'dice.models',
    'dice.models.parameters',
    'dice.models.profiles',
    'dice.models.results',
    'dice.utils',
    'dice.utils.axes',
    'dice.utils.converters',
    'dice.utils.validators',
    'dice.utils.legacy_compatibility',
    'dice.visualization',
    'dice.visualization.plots',
    'dice.visualization.histograms',
    'dice.cli',
    'dice.cli.main',
    'dice.cli.arguments',
    # DICE GUI packages
    'dice_gui',
    'dice_gui.dice_gui',
    'dice_gui.dice_interface',
    'dice_gui.validators',
    'dice_gui.validation_manager',
    'dice_gui.styles',
    'dice_gui.accessibility',
    'dice_gui.proximity_widget',
    'dice_gui.presets',
    'dice_gui.collapsible_group',
    # Scientific packages
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.stats',
    'scipy.special',
    'pandas',
    'statsmodels',
    'statsmodels.api',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_qtagg',
    'seaborn',
    'joblib',
    # PyQt6 modules
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
]

a = Analysis(
    [str(root_path / 'dice_gui' / 'dice_gui.py')],
    pathex=[str(root_path)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'pdb',
        'flask',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_file,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_file,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=app_name,
    )

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe if onefile else coll,
        name=f'{app_name}.app',
        icon=icon_file,
        bundle_identifier='com.thiebes.dice',
        info_plist={
            'CFBundleName': app_name,
            'CFBundleDisplayName': 'DICE - Diffusion Insight Computation Engine',
            'CFBundleVersion': '1.3.0',
            'CFBundleShortVersionString': '1.3.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        },
    )
