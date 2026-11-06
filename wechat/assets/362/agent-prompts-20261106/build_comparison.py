#!/usr/bin/env python3
"""Assemble the complete comparison; preserve source text inside fenced blocks."""
from pathlib import Path
import re,os
BASE=Path(__file__).resolve().parent;DOC=BASE.parent
WB=Path('/home/lmxxf/work/workbuddy-study')
def rebase(text,folder):
 out=[];fence=None
 for line in text.splitlines():
  m=re.match(r'^\s*(`{3,}|~{3,})',line)
  if m:
   z=m[1]
   if fence is None:fence=z
   elif z[0]==fence[0] and len(z)>=len(fence):fence=None
   out.append(line);continue
  if fence is None:
   def sub(m):
    v=m[1]
    if v.startswith(('http:','https:','/','#','mailto:')):return m[0]
    p=(folder/v).resolve()
    return ']('+os.path.relpath(p,DOC)+')'
   line=re.sub(r'\]\(([^)]+)\)',sub,line)
  out.append(line)
 return '\n'.join(out)+'\n'
a=[(BASE/'comparison-intro.md').read_text()]
for title,file in [('A. Antigravity CLI：完整逆向记录',BASE/'agy/findings.md'),('B. Codex CLI：完整逆向记录',BASE/'codex/findings.md'),('C. Claude Code：完整逆向记录',DOC/'claude压缩与skill.md')]:
 a+=['\n\n---\n\n## '+title+'\n\n',rebase(file.read_text(),file.parent)]
# Include the entire new WorkBuddy appendix, not the historic unrelated research.
wbtext=(WB/'README.md').read_text().split('## 三、2026-11-06 增补：上下文压缩与 Skill Creator 全文档案\n',1)[1]
a+=['\n\n---\n\n## D. WorkBuddyAI：完整逆向记录\n\n',rebase(wbtext,WB)]
# /init full original is an auxiliary creation entry; include rather than only linking.
a+=['\n\n---\n\n## E. Claude新版 /init 完整原文及技能阶段中译\n\n以下含CLAUDE.md、hooks等相邻阶段，完整保留以便看技能创建的上下文；它受开关控制，不代表每次/init默认路径。\n\n````text\n',(BASE/'claude/new-init.txt').read_text(),'\n````\n\n### Phase6技能阶段中译\n\n',(BASE/'claude/new-init-skill.zh.txt').read_text()]
a+=['\n\n## 核验与交付边界\n\n各家安装文件均未修改，未触发模型总结或技能创建请求；没有修改公众号正文，没有提交Git。原始英文切片／JSON解码结果均由安装包取证；译文与评价为本次整理。静态未覆盖服务端隐藏prompt、账号实验分组、动态hook内容，不能承诺提取不可见内容。四家manifest与提取脚本可独立复核。\n']
out=DOC/'四大Agent压缩与Skill提炼横向对比.md'
out.write_text('\n'.join(line.rstrip() for line in ''.join(a).split('\n')))
print(out, out.stat().st_size, 'bytes', len(out.read_text().splitlines()),'lines')
