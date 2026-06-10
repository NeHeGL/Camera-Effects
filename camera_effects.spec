# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Camera Effects Playground
# 2026 Jeff Molofee (NeHe)
#
# Build:  pyinstaller camera_effects.spec --clean --noconfirm
# Output: dist/CameraEffects.exe  (single-file, windowed)

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Bundle mediapipe data files (models, etc.)
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['camera_effects.py'],
    pathex=[],
    binaries=collect_dynamic_libs('mediapipe'),
    datas=[
        # The selfie segmenter model (downloaded by the workflow before building)
        ('selfie_segmenter.tflite', '.'),
        # effects_lib is imported as a regular module — PyInstaller picks it up
        # automatically, but list it explicitly to be safe
        ('effects_lib.py', '.'),
    ] + mediapipe_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
        'mediapipe.python',
        'mediapipe.python.solutions',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CameraEffects',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
