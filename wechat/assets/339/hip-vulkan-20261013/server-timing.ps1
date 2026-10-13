param([ValidateSet('hip','vulkan')][string]$Backend)
$ErrorActionPreference='Stop'
$root='C:\LLama-Compare-20261013';$tag="server-$Backend"
$env:PATH="$root\rocm-portable\_rocm_sdk_libraries\bin;$root\rocm-portable\_rocm_sdk_core\bin;"+$env:PATH
if(@(Get-Process -Name 'llama-server','llama-bench' -ErrorAction SilentlyContinue).Count){throw 'model process present'}
$device='Vulkan1';if($Backend -eq 'hip'){$device='ROCm0'}
$argv=@('-m',"$root\models\Qwen3-14B-Q4_K_M.gguf",'--device',$device,'--split-mode','none','-ngl','999','--fit','off','-c','4096','-b','512','-ub','256','-t','8','-fa','on','-ctk','f16','-ctv','f16','-np','1','--host','127.0.0.1','--port','18089','--no-webui','--cache-ram','0')
$p=Start-Process "$root\$Backend\llama-server.exe" -ArgumentList $argv -WorkingDirectory "$root\$Backend" -PassThru -RedirectStandardOutput "$root\$tag.out.log" -RedirectStandardError "$root\$tag.err.log"
@{pid=$p.Id;args=$argv}|ConvertTo-Json -Depth 4|Set-Content "$root\$tag-launch.json"
$results=@()
try{
 $watch=[Diagnostics.Stopwatch]::StartNew();$ready=$false
 while(-not $p.HasExited -and $watch.Elapsed.TotalSeconds -lt 180){try{$h=Invoke-RestMethod 'http://127.0.0.1:18089/health' -TimeoutSec 2;if($h.status -eq 'ok'){$ready=$true;break}}catch{};Start-Sleep -Seconds 2}
 if(-not $ready){throw 'not ready'}
 $text=('The patient programmer examines a numerical experiment and records every result carefully. '*500)
 $tokens=(Invoke-RestMethod 'http://127.0.0.1:18089/tokenize' -Method Post -ContentType 'application/json' -Body (@{content=$text;add_special=$false}|ConvertTo-Json)).tokens
 # One warm request per length, then three measured requests per length; each request discards prefix cache.
 foreach($round in 0,1,2,3){foreach($length in 512,2048){
  $body=@{prompt=@($tokens[0..($length-1)]);n_predict=128;temperature=0;seed=123;cache_prompt=$false;ignore_eos=$true;stream=$true}|ConvertTo-Json -Compress -Depth 4
  $req=[Net.HttpWebRequest]::Create('http://127.0.0.1:18089/completion');$req.Method='POST';$req.ContentType='application/json';$req.Timeout=120000;$req.ReadWriteTimeout=120000
  $bytes=[Text.Encoding]::UTF8.GetBytes($body);$req.ContentLength=$bytes.Length
  $w=[Diagnostics.Stopwatch]::StartNew();$os=$req.GetRequestStream();$os.Write($bytes,0,$bytes.Length);$os.Close()
  $resp=$req.GetResponse();$reader=New-Object IO.StreamReader($resp.GetResponseStream());$first=$null;$final=$null;$events=0
  try{while(-not $reader.EndOfStream){$line=$reader.ReadLine();if($line.StartsWith('data: ')){$payload=$line.Substring(6);if($payload -eq '[DONE]'){continue};$j=$payload|ConvertFrom-Json;$events++;if($null -eq $first -and $j.content.Length -gt 0){$first=$w.Elapsed.TotalMilliseconds};if($j.stop){$final=$j}}}}finally{$reader.Dispose();$resp.Dispose()}
  $results+=@{backend=$Backend;round=$round;warmup=($round -eq 0);prompt_tokens=$length;n_predict=128;ttft_ms=$first;total_ms=$w.Elapsed.TotalMilliseconds;events=$events;final=$final}
  $results|ConvertTo-Json -Depth 10|Set-Content "$root\$tag-timings.json"
 }}
}finally{if(-not $p.HasExited){Stop-Process -Id $p.Id;$p.WaitForExit(10000)|Out-Null};@{pid=$p.Id;exit=$p.ExitCode;requests=$results.Count}|ConvertTo-Json|Set-Content "$root\$tag-receipt.json"}
