$ErrorActionPreference='Stop'
$servers=@(Get-CimInstance Win32_Process | Where-Object {
 $_.Name -eq 'python.exe' -and $_.ExecutablePath -like 'C:\Strata-Lab\*' -and $_.CommandLine -match 'serve[\\/]server.py'
})
$ids=@($servers.ProcessId)
foreach($p in $servers){
 if($ids -notcontains $p.ParentProcessId){ & taskkill.exe /PID $p.ProcessId /T /F }
}
