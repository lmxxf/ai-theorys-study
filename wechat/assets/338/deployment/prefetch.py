import sys,pathlib,os,time
os.environ['HTTPS_PROXY']='http://127.0.0.1:17899'
os.environ['HTTP_PROXY']=os.environ['HTTPS_PROXY']
sys.path.insert(0,r'C:\Strata-Lab\Strata')
import setup
fam=setup.FAMILIES['qwen']; name=setup.model_file(fam,'Q2_0',2)
staging=pathlib.Path(r'C:\Strata-Lab\prefetch')/name
setup.download(fam['hf'].format(q='Q2_0')+name,staging,name)
# Keep prefetched data separate; the controller moves it after checking the main install's phase.
print('PREFETCH_COMPLETE',staging,flush=True)
