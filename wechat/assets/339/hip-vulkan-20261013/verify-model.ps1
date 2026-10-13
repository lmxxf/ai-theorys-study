$ErrorActionPreference='Stop'
$root='C:\LLama-Compare-20261013'
$f=Get-Item "$root\models\Qwen3-14B-Q4_K_M.gguf"
$h=(Get-FileHash $f.FullName).Hash.ToLower()
if($f.Length -ne 9001752960 -or $h -ne '500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0'){throw 'model size/hash mismatch'}
@{path=$f.FullName;bytes=$f.Length;sha256=$h;utc=(Get-Date).ToUniversalTime().ToString('o')}|ConvertTo-Json|Set-Content "$root\model-receipt.json"
Get-Content "$root\model-receipt.json"
