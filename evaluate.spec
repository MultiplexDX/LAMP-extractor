# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# datas = [('b02_lamp_national_testing_02102022-03112022', './b02_lamp_national_testing_02102022-03112022')]
datas = []
datas += collect_data_files('lamp_extractor')

block_cipher = None


a = Analysis(
    ['evaluate.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

options = [('W ignore', None, 'OPTION') ]

exe = EXE(
    pyz,
    a.scripts,
    options,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='evaluate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
