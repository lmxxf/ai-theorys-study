$ErrorActionPreference='Stop'
$root='C:\LLama-Compare-20261013'
$items=Get-Content "$root\rocm-receipts.json" -Raw|ConvertFrom-Json
Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force "$root\rocm-portable"|Out-Null
foreach($f in $items){
 $path=Join-Path $root $f.filename
 if((Get-Item $path).Length -ne $f.bytes -or (Get-FileHash $path).Hash.ToLower() -ne $f.sha256){throw "bad wheel $path"}
 $zip=[IO.Compression.ZipFile]::OpenRead($path)
 try { foreach($entry in $zip.Entries){
  $dest=Join-Path "$root\rocm-portable" $entry.FullName
  if($entry.Name -eq ''){New-Item -ItemType Directory -Force $dest|Out-Null;continue}
  New-Item -ItemType Directory -Force (Split-Path $dest)|Out-Null
  [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$dest,$true)
 }} finally {$zip.Dispose()}
}
$env:PATH="$root\rocm-portable\_rocm_sdk_libraries\bin;$root\rocm-portable\_rocm_sdk_core\bin;"+$env:PATH
$p=Start-Process "$root\hip\llama-cli.exe" -ArgumentList '--list-devices' -WorkingDirectory "$root\hip" -Wait -PassThru -RedirectStandardOutput "$root\hip-ready.out.log" -RedirectStandardError "$root\hip-ready.err.log"
"exit $($p.ExitCode)"
Get-Content "$root\hip-ready.out.log","$root\hip-ready.err.log"
