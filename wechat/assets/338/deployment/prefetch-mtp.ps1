$ErrorActionPreference = 'Stop'
$env:HTTPS_PROXY='http://127.0.0.1:17899'
$env:HTTP_PROXY=$env:HTTPS_PROXY
$env:NO_PROXY='127.0.0.1,localhost'
Set-Location C:\Strata-Lab\Strata
& '.\.venv\Scripts\python.exe' -u tools\mtp_fetch.py fetch --out C:\Strata-Lab\Strata-data\mtp
Set-Content C:\Strata-Lab\mtp-fetch-exit.txt $LASTEXITCODE
exit $LASTEXITCODE
