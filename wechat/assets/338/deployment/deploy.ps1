$ErrorActionPreference = 'Stop'
$root = 'C:\Strata-Lab'
$env:HTTPS_PROXY = 'http://127.0.0.1:17899'
$env:HTTP_PROXY = $env:HTTPS_PROXY
$env:NO_PROXY = '127.0.0.1,localhost'
$env:PYTHONUNBUFFERED = '1'
$python = "$root\Python\python.exe"
if (!(Test-Path $python)) {
    & curl.exe --fail --location --retry 3 --proxy $env:HTTPS_PROXY 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -o "$root\python-setup.exe"
    if ($LASTEXITCODE -ne 0) { throw 'Python download failed' }
    $p = Start-Process "$root\python-setup.exe" -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=0','Include_launcher=0','Include_test=0',"TargetDir=$root\Python" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python installer exit $($p.ExitCode)" }
}
Set-Location "$root\Strata"
if (!(Test-Path '.venv\Scripts\python.exe')) { & $python -m venv .venv }
& '.\.venv\Scripts\python.exe' -u setup.py --yes --family qwen --model Q2_0 --backend hip --context 4096 --vision no --data-dir "$root\Strata-data" --no-start
$code = $LASTEXITCODE
Set-Content "$root\setup-exit.txt" $code
exit $code
