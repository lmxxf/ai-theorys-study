#!/usr/bin/env python3
"""Recount the article snapshot and optionally check FFX's Windows x64 layout.

python3 wechat/assets/309/verify_sources.py
python3 wechat/assets/309/verify_sources.py --sdk-headers /path/to/ffx_api
The optional directory must contain official ffx_api.h, ffx_api_types.h,
and ffx_upscale.h. No network access or GPU execution is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1] / '297')
parser.add_argument('--sdk-headers', type=Path)
args = parser.parse_args()
repo = args.repo.resolve()

def git(*arguments):
    return subprocess.check_output(['git', '-C', str(repo), *arguments]).decode()

paths = [Path(s) for s in git('ls-files', 'src', 'shaders', 'scripts', 'tools', 'Development').splitlines()]
result = {'commit': git('rev-parse', 'HEAD').strip(), 'directories': {}}
for directory in ('src', 'shaders', 'scripts', 'tools', 'Development'):
    files = [p for p in paths if p.parts[0] == directory]
    entry = {'tracked_files': len(files)}
    if directory != 'Development':
        entry['physical_lines'] = sum((repo / p).read_bytes().count(b'\n') for p in files)
    result['directories'][directory] = entry
result['source_lines'] = sum(result['directories'][d]['physical_lines'] for d in ('src', 'shaders'))
result['total_lines'] = sum(result['directories'][d]['physical_lines'] for d in ('src', 'shaders', 'scripts', 'tools'))
bench = (repo / 'scripts/bench.ps1').read_text()
result['bench'] = {'lines': len(bench.splitlines()), 'bytes': (repo / 'scripts/bench.ps1').stat().st_size,
                   'FAST_PATH_occurrences': bench.count('FAST PATH'),
                   'NULL_RESULT_occurrences': bench.count('NULL RESULT'),
                   'FIX_comment_headers': len(re.findall(r'^\s*# FIX:', bench, re.M))}
shader_texts = [(repo / p).read_text() for p in paths if p.parts[0] == 'shaders']
result['H'] = {'files_defining': sum(bool(re.search(r'float\s+H\s*\(', t)) for t in shader_texts),
               'definition_and_call_tokens': sum(len(re.findall(r'\bH\s*\(', t)) for t in shader_texts)}
result['wrapper_lines'] = sum((repo / p).read_bytes().count(b'\n') for p in ('src/native_pso.h', 'src/native_pinned_resource.h'))
result['sha256'] = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in (
    'src/native_submission_order_probe.cpp', 'src/native_c64.h', 'src/native_pinned_resource.h',
    'src/native_pso.h', 'shaders/native_wave_qkv.hlsl', 'shaders/native_sat_cast.hlsli', 'scripts/bench.ps1')}

if args.sdk_headers:
    sdk = args.sdk_headers.resolve()
    fields = {'commandList': 16, 'color': 24, 'depth': 72, 'motionVectors': 120,
              'exposure': 168, 'reactive': 216, 'transparencyAndComposition': 264,
              'output': 312, 'jitterOffset': 360, 'motionVectorScale': 368,
              'renderSize': 376, 'upscaleSize': 384, 'enableSharpening': 392,
              'sharpness': 396, 'frameTimeDelta': 400, 'preExposure': 404, 'reset': 408}
    source = '#include <cstddef>\n#include "ffx_upscale.h"\n'
    source += 'static_assert(sizeof(void*) == 8);\nstatic_assert(sizeof(ffxApiHeader) == 16);\n'
    source += 'static_assert(sizeof(FfxApiResource) == 48);\n'
    for field, offset in fields.items():
        source += f'static_assert(offsetof(ffxDispatchDescUpscale, {field}) == {offset}, "{field}");\n'
    with tempfile.TemporaryDirectory(prefix='wechat309-abi-') as scratch:
        p = Path(scratch) / 'layout.cpp'
        p.write_text(source)
        subprocess.run(['x86_64-w64-mingw32-g++', '-std=c++17', '-I', str(sdk),
                        '-c', str(p), '-o', str(p.with_suffix('.o'))], check=True)
    result['windows_x64_official_layout'] = fields
    result['official_header_sha256'] = {n: hashlib.sha256((sdk / n).read_bytes()).hexdigest()
                                        for n in ('ffx_api.h', 'ffx_api_types.h', 'ffx_upscale.h')}
print(json.dumps(result, ensure_ascii=False, indent=2))
