# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for the DICE GUI application.

Build from the repository root, in an environment synced from uv.lock with
PyInstaller installed from .github/pyinstaller-requirements.txt (see
.github/workflows/build-release.yml, which syncs without extras so that test
and web-app packages are not bundled):

    pyinstaller dice-gui.spec

Output:
    dist/DICE.exe (Windows, one file)
    dist/DICE (Linux, one file)
    dist/DICE.app (macOS, one-folder application bundle)
"""

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

app_name = 'DICE'
root_path = Path(SPECPATH)

# One file on Windows and Linux. On macOS, PyInstaller deprecates one-file
# .app bundles (an error from PyInstaller 7) because they conflict with code
# signing and Gatekeeper, so the bundle is built in one-folder mode.
onefile = sys.platform != 'darwin'

with open(root_path / 'dice' / '__init__.py') as _f:
    _version_match = re.search(r"__version__\s*=\s*['\"](.+?)['\"]", _f.read())
    app_version = _version_match.group(1) if _version_match else '0.0.0'

# PyInstaller ignores icons on Linux, and macOS would need an .icns file.
if sys.platform == 'win32':
    icon_file = str(root_path / 'webapp' / 'static' / 'favicon.ico')
else:
    icon_file = None

# The entry script is dice_gui/dice_gui.py, not dice_gui/__main__.py (the
# `dice-gui` console script): __main__.py starts joblib workers at launch to
# warm them up, and in a frozen application a loky worker re-runs the entry
# script instead of the worker code, so each launch would start further
# copies of the application.

# The project is built from the source tree, not installed, so put it on
# sys.path before collect_submodules imports the packages in a subprocess.
sys.path.insert(0, str(root_path))

# Static analysis follows the GUI's direct imports, but dice/__init__.py and
# dice/core/__init__.py resolve their public names lazily through
# importlib.import_module (PEP 562), which PyInstaller cannot follow. Bundle
# every first-party submodule, so a module reached only through those lazy
# names is still present. on_error='raise' fails the build if a package
# cannot be imported, instead of silently bundling a partial list.
first_party_modules = [
    *collect_submodules('dice', on_error='raise'),
    *collect_submodules('dice_gui', on_error='raise'),
]
for _required in ('dice.core.fitting', 'dice_gui.dice_gui'):
    if _required not in first_party_modules:
        raise SystemExit(
            f"collect_submodules did not find {_required!r}; "
            "run PyInstaller from the repository root"
        )

hiddenimports = [
    *first_party_modules,
    # dice.core re-exports fft_cnr lazily, by name, in its lazy-import table,
    # so keep it bundled even if the direct imports of it change.
    *collect_submodules('fft_cnr', on_error='raise'),
    # The Matplotlib hook bundles only the backends named in matplotlib.use()
    # calls (Agg, in dice/visualization/histograms.py). If pyplot resolves its
    # backend while a QApplication is running and none has been set, it
    # selects QtAgg, so bundle that backend as well.
    'matplotlib.backends.backend_qtagg',
]

# Example parameter files, which the application finds in sys._MEIPASS
# (dice.utils.runtime.examples_directory) for the GUI's example menu. The
# menu falls back to embedded copies (dice_gui/example_parameters.py) when it
# cannot find them, as in a wheel installation.
datas = [
    (str(root_path / 'data'), 'data'),
]

a = Analysis(
    [str(root_path / 'dice_gui' / 'dice_gui.py')],
    pathex=[str(root_path)],
    binaries=[],
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
    noarchive=False,
)

pyz = PYZ(a.pure)

# UPX is off: it is applied whenever a upx binary happens to be on PATH (the
# Ubuntu runner image installs one), which would let an unpinned tool rewrite
# the release binaries, and UPX-compressed Qt libraries are a known source of
# crashes.
if onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
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
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
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
        a.datas,
        strip=False,
        upx=False,
        name=app_name,
    )

if sys.platform == 'darwin':
    app = BUNDLE(
        exe if onefile else coll,
        name=f'{app_name}.app',
        icon=icon_file,
        bundle_identifier='com.thiebes.dice',
        info_plist={
            'CFBundleName': app_name,
            'CFBundleDisplayName': 'DICE - Diffusion Insight Computation Engine',
            'CFBundleVersion': app_version,
            'CFBundleShortVersionString': app_version,
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        },
    )
