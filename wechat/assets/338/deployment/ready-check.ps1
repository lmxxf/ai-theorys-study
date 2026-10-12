$ErrorActionPreference='Stop'
$root='C:\Strata-Lab'
$name='Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf'
foreach($suffix in @('', '.done')) {
 $src="$root\prefetch\$name$suffix"
 if(Test-Path $src){Move-Item $src "$root\Strata-data\models\Q2_0\"}
}
Get-Item "$root\Strata-data\models\Q2_0\$name" | Select-Object Name,Length
'GPU_LOCK:'
Test-Path D:\DLSSNR-Lab\gpu.lock
if(Test-Path D:\DLSSNR-Lab\gpu.lock){Get-Content D:\DLSSNR-Lab\gpu.lock}
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'Stellar|SB-Win|Onimusha|Cyberpunk|strata|hip|rtc|bench' } | Select-Object Name,ProcessId,CommandLine | Format-List
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory | Format-List
