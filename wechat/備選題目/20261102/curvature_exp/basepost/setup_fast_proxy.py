"""Create a private, temporary download-only proxy; never edits the running global config."""
import os,yaml,json
from pathlib import Path
source=yaml.safe_load(Path('/tmp/ShellCrash/config.yaml').read_text())
allnodes=source['proxies'];nodes=[]
for term in ['Japan','USA Seattle 01','Hong Kong 01','Singapore 12']:
 match=next((p for p in allnodes if term in p['name']),None)
 if match and match not in nodes:nodes.append(match)
assert nodes
folder=Path('/tmp/basepost-fast');folder.mkdir(mode=0o700,exist_ok=True)
config={'mixed-port':18992,'bind-address':'127.0.0.1','allow-lan':False,'external-controller':'127.0.0.1:18993','mode':'rule','log-level':'error','ipv6':False,'routing-mark':source.get('routing-mark',7894),'interface-name':'wlP9s9','proxies':nodes,'proxy-groups':[{'name':'Fast','type':'select','proxies':[p['name'] for p in nodes]}],'rules':['MATCH,Fast'],'dns':{'enable':True,'listen':'127.0.0.1:18994','enhanced-mode':'fake-ip','fake-ip-range':'198.19.0.1/16','default-nameserver':['223.5.5.5','119.29.29.29'],'nameserver':['223.5.5.5','119.29.29.29'],'proxy-server-nameserver':['223.5.5.5','119.29.29.29']}}
p=folder/'config.yaml';p.write_text(yaml.safe_dump(config,allow_unicode=True));p.chmod(0o600)
print(json.dumps({'candidates':[p['name'] for p in nodes],'private_config':str(p)},ensure_ascii=False))
