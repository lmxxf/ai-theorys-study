"""Resume the authorized paper experiment after collection, independently of the chat connection."""
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path('/home/lmxxf/work/strata-papers-study')
OLD = pathlib.Path('/home/lmxxf/work/strata-route-study')
REMOTE = 'C:/Strata-Lab/papers-speed-study'
STATE = ROOT / 'pipeline-state.json'
state = json.loads(STATE.read_text()) if STATE.exists() else {'completed': []}

def save(stage, error=None):
    state.update(stage=stage, error=error, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)
    print(state['updated'], stage, error or '', flush=True)

def run(args, log=None):
    if log:
        with pathlib.Path(log).open('w') as stream:
            subprocess.run(list(map(str, args)), check=True, stdout=stream, stderr=subprocess.STDOUT)
    else:
        subprocess.run(list(map(str, args)), check=True)

def py(script, *args, log=None):
    run([sys.executable, script, *args], log)

def ssh(command, capture=False):
    return subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'amd9070', command],
                          check=True, text=True, capture_output=capture)

def fetch(name, local):
    run(['scp', 'amd9070:' + REMOTE + '/' + name, local])

def stage(name, action):
    if name in state['completed']:
        return
    save(name)
    action()
    state['completed'].append(name)
    save(name + ': complete')

def wait_capture():
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        result = ssh('powershell -NoProfile -Command "Get-Content C:\\Strata-Lab\\papers-study\\collection.log -Tail 1"', True)
        line = result.stdout.strip()
        print(line, flush=True)
        if line == 'COLLECTION_COMPLETE 580':
            return
        if 'Traceback' in line or 'Error' in line:
            raise RuntimeError(line)
        time.sleep(30)
    raise TimeoutError('Collection did not complete within two hours')

def sync():
    py(HERE / 'sync_capture.py', ROOT, '--complete')
    records = json.loads((ROOT / 'collection.json').read_text())
    assert len(records) == len({r['id'] for r in records}) == 580
    assert len({r['paper_id'] for r in records if r['split'] == 'train'}) == 30
    for name in ['prompts.json', 'collection.log']:
        run(['scp', 'amd9070:C:/Strata-Lab/papers-study/' + name, ROOT / name])

def deploy():
    ssh('powershell -NoProfile -Command "New-Item -ItemType Directory -Force C:\\Strata-Lab\\papers-speed-study | Out-Null; Copy-Item C:\\Strata-Lab\\speed-study\\base-config.json,C:\\Strata-Lab\\speed-study\\cases.json,C:\\Strata-Lab\\speed-study\\lora-runtime.bin C:\\Strata-Lab\\papers-speed-study"')
    for local, name in [(ROOT / 'speed/lora-runtime.bin', 'new-lora-runtime.bin'),
                        (ROOT / 'speed/lora-fixture.bin', 'lora-fixture.bin'),
                        (HERE.parent / 'route-speed/benchmark.py', 'benchmark.py')]:
        run(['scp', local, 'amd9070:' + REMOTE + '/' + name])

BENCH = ('C:\\Strata-Lab\\Strata\\.venv\\Scripts\\python.exe '
         'C:\\Strata-Lab\\papers-speed-study\\benchmark.py '
         '--output-dir C:\\Strata-Lab\\papers-speed-study '
         '--new-lora C:\\Strata-Lab\\papers-speed-study\\new-lora-runtime.bin')

def pilot():
    ssh(BENCH + ' --pilot')
    fetch('lora-fixture.bin.out', ROOT / 'speed/lora-fixture.bin.out')
    fetch('pilot-D-engine.log', ROOT / 'speed/pilot-D-engine.log')
    import numpy as np
    expected = np.load(ROOT / 'speed/lora-fixture-expected.npy')
    actual = np.fromfile(ROOT / 'speed/lora-fixture.bin.out', dtype='<f4').reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-4)
    text = (ROOT / 'speed/pilot-D-engine.log').read_text(errors='replace')
    import re
    checks = re.findall(r'verified (\d+) ok / (\d+) BAD', text)
    if not checks or max(int(ok) for ok, bad in checks) == 0 or any(int(bad) for ok, bad in checks):
        raise RuntimeError('Inspect pilot copy verification before timing: ' + str(checks[-5:]))
    (HERE / 'runtime-validation.json').write_text(json.dumps({
        'max_absolute_error': float(np.max(np.abs(actual - expected))),
        'atol': 2e-4, 'rtol': 1e-4, 'copy_verification': checks}, indent=2))

def timed():
    exists = ssh('powershell -NoProfile -Command "Test-Path C:\\Strata-Lab\\papers-speed-study\\results.json"', True).stdout.strip() == 'True'
    ssh(BENCH + (' --resume' if exists else ''))
    fetch('results.json', ROOT / 'speed/results.json')
    py(HERE.parent / 'route-speed/analyze.py', ROOT / 'speed/results.json', HERE / 'speed', '--arms', 'ABCD')

try:
    stage('wait_collection', wait_capture)
    stage('sync', sync)
    stage('dataset', lambda: py(HERE.parent / 'route-lora/build_dataset.py', ROOT, '--skip-windows', '1', log=ROOT / 'dataset-build.log'))
    stage('train', lambda: py(HERE.parent / 'route-lora/train.py', ROOT, '--objective', 'uncached', '--base', 'next', '--output', 'trained-cold', log=ROOT / 'training-cold.log'))
    stage('groups', lambda: py(HERE.parent / 'route-lora/evaluate_groups.py', ROOT, '--run', 'trained-cold', '--base', 'next', '--uncached'))
    stage('compare', lambda: py(HERE / 'compare_models.py', ROOT, OLD / 'trained-cold', ROOT / 'trained-cold', HERE / 'comparison.json'))
    stage('export', lambda: py(HERE.parent / 'route-lora/export_cold.py', '--study', ROOT, '--destination', HERE))
    stage('runtime', lambda: py(HERE.parent / 'route-speed/prepare_runtime.py', ROOT))
    stage('deploy', deploy)
    stage('pilot', pilot)
    stage('speed', timed)
    save('COMPLETE')
except Exception as exc:
    save(state['stage'], repr(exc))
    raise
