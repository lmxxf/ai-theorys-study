$ErrorActionPreference='Stop'
$root='C:\LLama-Compare-20261013'
$env:PATH="$root\rocm-portable\_rocm_sdk_libraries\bin;$root\rocm-portable\_rocm_sdk_core\bin;"+$env:PATH
$sequence=@('hip','vulkan','vulkan','hip','hip','vulkan')
$receipts=@()
for($i=0;$i -lt $sequence.Count;$i++){
 $backend=$sequence[$i];$tag=('formal-{0:D2}-{1}' -f $i,$backend)
 $conflict=@(Get-CimInstance Win32_Process|Where-Object {$_.Name -match 'StellarBlade|SB-Win64|Onimusha|re2\.exe|re4\.exe|Cyberpunk2077|MonsterHunter|strata|llama-server|llama-bench'})
 if($conflict.Count){throw 'conflicting process; bench stopped'}
 $device='Vulkan1';if($backend -eq 'hip'){$device='ROCm0'}
 $argv=@('-m',"$root\models\Qwen3-14B-Q4_K_M.gguf",'-dev',$device,'-sm','none','-ngl','999','-p','512,2048','-n','128','-b','512','-ub','256','-fa','on','-ctk','f16','-ctv','f16','-t','8','-r','3','-o','json')
 $start=(Get-Date).ToUniversalTime()
 $p=Start-Process "$root\$backend\llama-bench.exe" -ArgumentList $argv -WorkingDirectory "$root\$backend" -PassThru -RedirectStandardOutput "$root\$tag.json" -RedirectStandardError "$root\$tag.err.log"
 $handle=$p.Handle
 @{pid=$p.Id;backend=$backend;args=$argv;start=$start.ToString('o')}|ConvertTo-Json -Depth 4|Set-Content "$root\$tag-launch.json"
 $elapsed=[Diagnostics.Stopwatch]::StartNew();$abort=$null
 while(-not $p.WaitForExit(5000)){
  $game=@(Get-Process|Where-Object {$_.ProcessName -match 'StellarBlade|SB-Win64|Onimusha|Cyberpunk2077|MonsterHunter'})
  if($game.Count -or $elapsed.Elapsed.TotalSeconds -gt 600){$abort='game or timeout';Stop-Process -Id $p.Id;$p.WaitForExit();break}
 }
 $p.WaitForExit()
 $receipts+=@{tag=$tag;backend=$backend;pid=$p.Id;exit=$p.ExitCode;abort=$abort;seconds=$elapsed.Elapsed.TotalSeconds;args=$argv}
 $receipts|ConvertTo-Json -Depth 5|Set-Content "$root\formal-receipts.json"
 if($p.ExitCode -ne 0 -or $abort){throw "$tag failed"}
 $content=Get-Content "$root\$tag.json" -Raw|ConvertFrom-Json
 if(@($content).Count -ne 3){throw 'unexpected result count'}
 Start-Sleep -Seconds 3
}
