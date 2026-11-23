"""Static Python AST counts: no code is imported or executed."""
import ast,csv,json,collections,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;totals=collections.defaultdict(collections.Counter);errors=[]
for r in csv.DictReader(gzip.open(ROOT/'inventory_files.tsv.gz','rt',encoding='utf-8'),delimiter='\t'):
 if r['extension']!='.py':continue
 p=BASE/r['repo']/r['path'];c=totals[r['repo']];c['python_files']+=1
 try:tree=ast.parse(p.read_text())
 except(SyntaxError,UnicodeError,ValueError)as e:errors.append(dict(repo=r['repo'],path=r['path'],error=type(e).__name__));continue
 c['parsed_python_files']+=1;doc_nodes=set();doc_lines=set();multiline_lines=set()
 for n in ast.walk(tree):
  if isinstance(n,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))and n.body:
   f=n.body[0]
   if isinstance(f,ast.Expr)and isinstance(f.value,ast.Constant)and isinstance(f.value.value,str):doc_nodes.add(id(f));doc_lines.update(range(f.lineno,f.end_lineno+1))
 for n in ast.walk(tree):
  if isinstance(n,ast.stmt)and id(n)not in doc_nodes:c['non_docstring_statement_nodes']+=1
  if isinstance(n,ast.Constant)and isinstance(n.value,str)and hasattr(n,'lineno')and n.end_lineno>n.lineno:multiline_lines.update(range(n.lineno,n.end_lineno+1))
 c['docstring_physical_lines']+=len(doc_lines);c['multiline_string_lines_outside_docstrings']+=len(multiline_lines-doc_lines)
(ROOT/'python_logic_summary.json').write_text(json.dumps(dict(metrics={k:dict(v)for k,v in totals.items()},parse_failures=errors,notes='AST statement nodes are structural counts, not cyclomatic complexity or quality; multiline strings may contain prompts, SQL, code templates, or fixtures. Non-Python lines not compared by this metric.'),ensure_ascii=False,indent=2));print(json.dumps({k:dict(v)for k,v in totals.items()},indent=2))
