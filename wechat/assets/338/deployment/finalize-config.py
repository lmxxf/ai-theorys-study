import pathlib,json,shutil
root=pathlib.Path(r'C:\Strata-Lab');p=root/'Strata'/'strata-q2_0.json'
backup=root/'installer-original-config.json'
if not backup.exists():shutil.copyfile(p,backup)
cfg=json.loads(p.read_text());cfg.setdefault('env',{})['STRATA_ROUTE_TAIL_SKIP']='0'
p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
print('Saved baseline: STRATA_ROUTE_TAIL_SKIP=0')
