$ErrorActionPreference='Stop'
$root='C:\Strata-Lab'
$files=@('speed-engine\strata.exe','speed-engine\amdhip64_7.dll','speed-study\lora-runtime.bin','speed-study\base-config.json','Strata-speed\src\core\expert_source.cpp','Strata-speed\src\core\verify.cpp','Strata-speed\src\core\lookahead_lora.hpp','Strata-speed\src\program\generate.cpp')
$hashes=@{}
foreach($relative in $files){$hashes[$relative]=(Get-FileHash (Join-Path $root $relative) -Algorithm SHA256).Hash}
$record=[ordered]@{
 timestamp=(Get-Date -Format o)
 source_commit=(& git -C "$root\Strata-speed" rev-parse HEAD)
 cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
 gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
 hashes=$hashes
}
$record | ConvertTo-Json -Depth 7 | Set-Content "$root\speed-study\native-provenance.json" -Encoding UTF8
