$ErrorActionPreference='Stop'
$root='C:\Strata-Lab'
$busy=Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'Stellar|SB-Win|Onimusha|Cyberpunk|strata.exe|hip_rtc|native.*bench' }
if($busy){$busy | Select-Object Name,ProcessId | Format-Table;throw 'GPU workload already running'}
$lockPath='D:\DLSSNR-Lab\gpu.lock'
$lock=$null
try {
 $lock=[IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
 $msg=[Text.Encoding]::UTF8.GetBytes("Strata baseline server; controller PID=$PID; $(Get-Date -Format o)")
 $lock.Write($msg,0,$msg.Length);$lock.Flush()
 Set-Location "$root\Strata"
 $env:STRATA_ROUTE_TAIL_SKIP='0'
 $env:PYTHONUNBUFFERED='1'
 $env:PYTHONIOENCODING='utf-8'
 & '.\.venv\Scripts\python.exe' -u serve\server.py --engine strata --config strata-q2_0.json --port 8080
 exit $LASTEXITCODE
} finally {
 if($null -ne $lock){$lock.Dispose();Remove-Item $lockPath}
}
