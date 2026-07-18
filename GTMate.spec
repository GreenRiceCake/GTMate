# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules


# Keep runtime assets such as bin/, models/, Updater.exe, curr_ver.json, and
# update_manifest.json outside this build. Inno Setup / release zip packaging
# should place those files beside GTMate.exe.
vosk_datas, vosk_binaries, vosk_hiddenimports = collect_all('vosk')
davey_datas, davey_binaries, davey_hiddenimports = collect_all('davey')
voice_recv_hiddenimports = collect_submodules('discord.ext.voice_recv')


a = Analysis(
    ['GTMate.py'],
    pathex=[],
    binaries=[
        *vosk_binaries,
        *davey_binaries,
    ],
    datas=[
        *vosk_datas,
        *davey_datas,
    ],
    hiddenimports=[
        *vosk_hiddenimports,
        *davey_hiddenimports,
        *voice_recv_hiddenimports,
        'vosk',
        'davey',
        'sounddevice',
        'discord.ext.voice_recv',
        'Crypto.Cipher.Salsa20',
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
    [],
    exclude_binaries=True,
    name='GTMate',
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
    icon=['GTMate.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GTMate',
)
