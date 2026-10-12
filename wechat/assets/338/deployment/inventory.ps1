$ErrorActionPreference='Stop'
$root='C:\Strata-Lab'
$info=[ordered]@{
 timestamp=(Get-Date -Format o)
 host_name=$env:COMPUTERNAME
 cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
 os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory
 memory=@(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,ConfiguredClockSpeed)
 disks=@(Get-Disk | Select-Object Number,FriendlyName,BusType,Size)
 volumes=@(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace)
 gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID)
 pagefile=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
 engine_build=Get-Content "$root\Strata\engine\BUILD.json" -Raw | ConvertFrom-Json
 engine_sha256=(Get-FileHash "$root\Strata\engine\strata.exe" -Algorithm SHA256).Hash
 source_commit=(& git -C "$root\Strata" rev-parse HEAD)
}
$info | ConvertTo-Json -Depth 8 | Set-Content "$root\inventory.json" -Encoding UTF8
Get-Content "$root\inventory.json"
