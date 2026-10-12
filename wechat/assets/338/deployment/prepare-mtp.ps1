$ErrorActionPreference='Stop'
$root='C:\Strata-Lab'
while (!(Test-Path "$root\mtp-fetch-exit.txt")) { Start-Sleep -Seconds 5 }
if ([int](Get-Content "$root\mtp-fetch-exit.txt") -ne 0) { throw 'MTP fetch failed' }
Set-Location "$root\Strata"
$env:STRATA_GGUF_PY="$root\Strata\third_party\llama.cpp\gguf-py"
& '.\.venv\Scripts\python.exe' -u tools\mtp_fetch.py verify --out "$root\Strata-data\mtp"
if ($LASTEXITCODE -ne 0) { throw 'MTP verification failed' }
& '.\.venv\Scripts\python.exe' -u tools\mtp_pack.py --src "$root\Strata-data\mtp" --experts q2_0 --out "$root\Strata-data\mtp\mtp-q2_0.gguf"
if ($LASTEXITCODE -ne 0) { throw 'MTP pack failed' }
& '.\.venv\Scripts\python.exe' -u tools\mtp_rt.py --gguf "$root\Strata-data\mtp\mtp-q2_0.gguf" --out "$root\Strata-data\mtp\rt"
$code=$LASTEXITCODE
Set-Content "$root\mtp-pack-exit.txt" $code
exit $code
