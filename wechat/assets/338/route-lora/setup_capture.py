import json,pathlib,zipfile,shutil
root=pathlib.Path(r'C:\Strata-Lab');eng=root/'capture-engine'
with zipfile.ZipFile(root/'dist-capture/strata-windows-x64-hip.zip') as z:z.extractall(eng)
for name in ['amdhip64_7.dll','amd_comgr.dll']:shutil.copyfile(eng/'rocm/bin'/name,eng/name)
c=json.loads((root/'config-before-64k.json').read_text());a=c['args']
for flag,val in [('--max-context','4096'),('--prefill','128'),('--short-read','65536'),('--prompt-cache','0'),('--vram-reserve-mib','4096')]:
 if flag in a:a[a.index(flag)+1]=val
 else:a.extend([flag,val])
c['exe']=str(eng/'strata.exe');c['lib_dirs']=[str(eng/'rocm/bin')];c['port']=18081
c['log']=str(root/'route-study/engine.log')
c['env'].update({'STRATA_CAPTURE_ROUTES':str(root/'route-study/routes.bin'),'STRATA_CAPTURE_STRIDE':'4','STRATA_CAPTURE_MAX_WINDOWS':'2048'})
(root/'route-study/config.json').write_text(json.dumps(c,indent=2))
print('capture configuration written')
