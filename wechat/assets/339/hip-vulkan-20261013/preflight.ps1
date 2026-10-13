param([ValidateSet('hip','vulkan')][string]$Backend)
$ErrorActionPreference='Stop'
$root='C:\LLama-Compare-20261013'
$tag="preflight-$Backend"
$games=@(Get-CimInstance Win32_Process|Where-Object {$_.Name -match 'StellarBlade|SB-Win64|Onimusha|re2\.exe|re4\.exe|Cyberpunk2077|MonsterHunter|strata|llama-server|llama-bench'})
if($games.Count){$games|Select-Object Name,ProcessId,CommandLine|ConvertTo-Json;throw 'conflicting game/model process'}
$gpu=@(Get-CimInstance Win32_VideoController|Where-Object Name -match '9070')
if($gpu.Count -ne 1 -or $gpu[0].DriverVersion -ne '32.0.31007.2048'){throw 'GPU/driver changed'}
if((Get-NetTCPConnection -LocalPort 18089 -State Listen -ErrorAction SilentlyContinue)){throw 'port in use'}
$device='Vulkan1'
if($Backend -eq 'hip'){$device='ROCm0';$env:PATH="$root\rocm-portable\_rocm_sdk_libraries\bin;$root\rocm-portable\_rocm_sdk_core\bin;"+$env:PATH}
$argv=@('-m',"$root\models\Qwen3-14B-Q4_K_M.gguf",'--device',$device,'--split-mode','none','-ngl','999','--fit','off','-c','4096','-b','512','-ub','256','-t','8','-fa','on','-ctk','f16','-ctv','f16','-np','1','--host','127.0.0.1','--port','18089','--no-webui','-lv','4')
$before=@((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage').CounterSamples|Select-Object Path,CookedValue)
$p=Start-Process "$root\$Backend\llama-server.exe" -ArgumentList $argv -WorkingDirectory "$root\$Backend" -PassThru -RedirectStandardOutput "$root\$tag.out.log" -RedirectStandardError "$root\$tag.err.log"
@{pid=$p.Id;args=$argv;utc=(Get-Date).ToUniversalTime().ToString('o');before=$before}|ConvertTo-Json -Depth 5|Set-Content "$root\$tag-launch.json"
$sw=[Diagnostics.Stopwatch]::StartNew();$samples=@();$ready=$false;$response=$null
try{
 while(-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt 180){
  try{$h=Invoke-RestMethod 'http://127.0.0.1:18089/health' -TimeoutSec 2;if($h.status -eq 'ok'){$ready=$true;break}}catch{}
  $samples+=@{elapsed=$sw.Elapsed.TotalSeconds;adapter=@((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage').CounterSamples|Select-Object Path,CookedValue)}
  Start-Sleep -Seconds 2
 }
 if($ready){
  $samples+=@{elapsed=$sw.Elapsed.TotalSeconds;ready=$true;process=@((Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage').CounterSamples|Where-Object InstanceName -match "pid_$($p.Id)_"|Select-Object Path,CookedValue)}
  $response=Invoke-RestMethod 'http://127.0.0.1:18089/completion' -Method Post -ContentType 'application/json' -Body '{"prompt":"Hello","n_predict":8,"temperature":0,"seed":123,"cache_prompt":false}' -TimeoutSec 90
  $samples+=@{elapsed=$sw.Elapsed.TotalSeconds;after_inference=$true;process=@((Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage').CounterSamples|Where-Object InstanceName -match "pid_$($p.Id)_"|Select-Object Path,CookedValue)}
 }
}finally{
 $naturalExit=$p.HasExited;if(-not $naturalExit){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue;$p.WaitForExit(10000)|Out-Null}
 @{ready=$ready;natural_exit=$naturalExit;exit_code=$p.ExitCode;pid=$p.Id;elapsed=$sw.Elapsed.TotalSeconds;samples=$samples;response=$response}|ConvertTo-Json -Depth 8|Set-Content "$root\$tag-receipt.json"
}
if(-not $ready){throw 'did not reach ready'}
