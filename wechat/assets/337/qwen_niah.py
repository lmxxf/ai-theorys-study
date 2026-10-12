#!/usr/bin/env python3
"""Paired retrieval pilot: official Qwen3-0.6B Base versus post-trained weights.

No training, no quantization, no context extension. Run this file to download
weights and run a small pilot. Increase --seeds for a larger experiment.
"""
import argparse
import gc
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=4, help='independent bindings; 4 is only a pilot')
    p.add_argument('--lengths', type=int, nargs='+', default=[4096, 16384])
    p.add_argument('--pairs', type=int, default=256)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'qwen-niah-results')
    p.add_argument('--prepare-only', action='store_true', help='build and validate prompts without loading weights')
    return p.parse_args()


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def extract_answer(completion, key):
    """Parse explicit answer formats without searching for the gold value.

    Accept a bare value, KEY: VALUE, or 'The value for KEY is VALUE'.
    Ignore empty template placeholders; reject conflicting concrete answers.
    """
    clean = completion.replace('**', '').replace('`', '')
    clean = re.sub(r'\\boxed\{\s*([A-Za-z0-9.]+)\s*\}', r'\1', clean)
    atom = r'([A-Za-z]+|[0-9]+(?:\.[0-9]+)?)'
    patterns = [
        rf'The value for {re.escape(key)} is\s*:?\s*{atom}[.!]?\s*',
        rf'{re.escape(key)}\s*:\s*{atom}[.!]?\s*',
        rf'Answer\s*:\s*{atom}[.!]?\s*',
        rf':?\s*{atom}[.!]?\s*',
    ]
    candidates = []
    for line in clean.splitlines():
        line = line.strip()
        for pattern in patterns:
            match = re.fullmatch(pattern, line, flags=re.IGNORECASE)
            if match:
                candidates.append(match.group(1))
                break
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def vocabulary(tok):
    # Keys and values occupy disjoint pools; each key and space-prefixed value
    # is one token. Validate the complete record too, rather than assuming BPE.
    words = []
    for i in range(10000, tok.vocab_size):
        w = tok.decode([i])
        if re.fullmatch(r'[A-Za-z]{3,12}', w) and encode(tok, w) == [i]:
            if len(encode(tok, ' ' + w)) == 1:
                words.append(w)
    words = sorted(set(words))
    random.Random(337).shuffle(words)
    middle = len(words) // 2
    return words[:middle], words[middle:]


def make_cases(tok, args):
    keys, values = vocabulary(tok)
    if min(len(keys), len(values)) < args.pairs:
        raise ValueError('Not enough verified single-token words')
    intro = ('Read the temporary key-value mappings below. Use these mappings, '
             'not general knowledge. For example, if a record says bird: stone, '
             'the value for bird is stone.\n\n')
    intro_ids = encode(tok, intro)
    # A transparent, artificial filler, not a claim to natural-document NIAH.
    filler_unit = encode(tok, 'Background text with no key-value mapping.\n')
    spacer = encode(tok, ' filler')
    newline = encode(tok, '\n')
    if len(spacer) != 1:
        raise ValueError('Expected a single-token filler')

    def filler(n):
        if n < 1:
            raise ValueError('Context too short for requested number of records')
        # Terminate at a record boundary; preserve exact token count.
        whole, remainder = divmod(n - 1, len(filler_unit))
        if whole and remainder == 0:
            whole -= 1
            remainder = len(filler_unit)
        return filler_unit * whole + spacer * remainder + newline

    cases = []
    for length in args.lengths:
        for seed in range(args.seeds):
            rng = random.Random(337000 + seed)
            kk = rng.sample(keys, args.pairs)
            vv = rng.sample(values, args.pairs)
            records = [encode(tok, f'{k}: {v}\n') for k, v in zip(kk, vv)]
            target = args.pairs // 2
            before = [t for row in records[:target] for t in row]
            body = [t for row in records for t in row]
            query = encode(tok, f'What is the value for {kk[target]}? '
                           f'Answer only with the value.\nThe value for {kk[target]} is')
            for residue in range(8):
                target_pos = (length // 2 // 8) * 8 + residue
                pre = target_pos - len(intro_ids) - len(before)
                post = length - len(intro_ids) - pre - len(body) - len(query)
                ids = intro_ids + filler(pre) + body + filler(post) + query
                text = tok.decode(ids)
                if len(ids) != length or encode(tok, text) != ids:
                    rebuilt = encode(tok, text)
                    at = next((i for i,(a,b) in enumerate(zip(ids, rebuilt)) if a != b), 0)
                    raise ValueError(f'Prompt round-trip failed at {at}: '
                                     f'{tok.decode(ids[max(0,at-5):at+8])!r}')
                if ids[target_pos] != encode(tok, kk[target])[0]:
                    raise ValueError('Target index verification failed')
                gold_ids = encode(tok, ' ' + vv[target])
                if encode(tok, text + ' ' + vv[target]) != ids + gold_ids:
                    raise ValueError('Gold continuation token boundary is unstable')
                cases.append(dict(case_id=f'{length}-{seed}-{residue}', seed=seed,
                                  length=length, residue=residue, target_pos=target_pos,
                                  key=kk[target], gold=vv[target], gold_id=gold_ids[0],
                                  input_ids=ids, text=text))
    return cases


def write_summary(rows, destination):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['protocol'], row['length'])].append(row)
    summary = []
    for (protocol, length), group in sorted(grouped.items()):
        by_residue = {str(r): sum(x['correct'] for x in group if x['residue'] == r)
                      / sum(x['residue'] == r for x in group) for r in range(8)}
        summary.append(dict(protocol=protocol, length=length, n=len(group),
                            accuracy=sum(x['correct'] for x in group) / len(group),
                            by_residue=by_residue))
    index = {(r['protocol'], r['case_id']): r for r in rows}
    paired = []
    for length in sorted({r['length'] for r in rows}):
        base = [r for r in rows if r['protocol'] == 'base_raw' and r['length'] == length]
        for other in ['post_raw', 'post_chat_no_thinking']:
            pairs = [(r, index[(other, r['case_id'])]) for r in base
                     if (other, r['case_id']) in index]
            if pairs:
                paired.append(dict(length=length, comparison=f'{other} vs base_raw',
                    n=len(pairs), post_only_correct=sum(not a['correct'] and b['correct'] for a,b in pairs),
                    base_only_correct=sum(a['correct'] and not b['correct'] for a,b in pairs)))
    result = dict(summary=summary, paired=paired,
        note='Exploratory pilot; offsets within a seed are correlated, not independent samples. '
             'Residues are reporting bins, not compression phases in these full-attention models.')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def protocol_input(tok, case, protocol):
    raw = case['input_ids']
    if protocol != 'post_chat_no_thinking':
        return raw, case['target_pos']
    ids = tok.apply_chat_template([dict(role='user', content=case['text'])],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    starts = [i for i in range(len(ids) - len(raw) + 1) if ids[i:i+len(raw)] == raw]
    if len(starts) != 1:
        raise ValueError('Chat template changed body tokenization')
    return ids, starts[0] + case['target_pos']


def main():
    args = arguments()
    if args.seeds < 1 or args.pairs < 2 or any(n < 1024 or n > 32000 for n in args.lengths):
        raise ValueError('Use positive seeds, at least two pairs, and lengths 1024..32000')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not args.prepare_only and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this experiment')
    names = ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-0.6B']
    tokenizers = [AutoTokenizer.from_pretrained(name) for name in names]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError('Tokenizer vocabularies differ; paired token comparison needs revision')
    cases = make_cases(tokenizers[0], args)
    for case in cases:
        if encode(tokenizers[1], case['text']) != case['input_ids']:
            raise ValueError('Post-trained tokenizer changes raw prompt tokenization')
        protocol_input(tokenizers[1], case, 'post_chat_no_thinking')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only and (args.output / 'predictions.jsonl').exists():
        raise FileExistsError('Results already exist. Choose a different --output directory.')
    serial = '\n'.join(json.dumps(c, ensure_ascii=False) for c in cases)
    (args.output / 'cases.jsonl').write_text(serial + '\n')
    meta = dict(models=names, torch=torch.__version__, transformers=transformers.__version__,
                seeds=args.seeds, lengths=args.lengths, pairs=args.pairs,
                cases_sha256=hashlib.sha256(serial.encode()).hexdigest(),
                dtype='bfloat16', attention='sdpa', max_new_tokens=16,
                scoring='format-normalized explicit answer; reject conflicting answers; case-sensitive gold match',
                filler='repeated neutral sentence; synthetic pilot, not paper replication')
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'Validated {len(cases)} paired prompts; output: {args.output}', flush=True)
    if args.prepare_only:
        return
    torch.manual_seed(337)
    rows = []
    with (args.output / 'predictions.jsonl').open('w') as out:
        for name, tok in zip(names, tokenizers):
            print(f'Loading {name}', flush=True)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                        attn_implementation='sdpa').to('cuda').eval()
            meta.setdefault('resolved_model_commits', {})[name] = model.config._commit_hash
            (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2))
            protocols = ['base_raw'] if name.endswith('-Base') else ['post_raw', 'post_chat_no_thinking']
            for protocol in protocols:
                for number, case in enumerate(cases, 1):
                    ids, target_pos = protocol_input(tok, case, protocol)
                    if len(ids) + 16 > 32768:
                        raise ValueError('Native context limit exceeded')
                    inputs = torch.tensor([ids], device='cuda')
                    with torch.inference_mode():
                        output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                            max_new_tokens=16, do_sample=False, use_cache=True,
                            pad_token_id=tok.eos_token_id, logits_to_keep=1)
                    completion = tok.decode(output[0, len(ids):], skip_special_tokens=True)
                    answer = extract_answer(completion, case['key'])
                    row = {k: v for k,v in case.items() if k not in ['text','input_ids','gold_id']}
                    row.update(protocol=protocol, actual_input_length=len(ids),
                               actual_target_pos=target_pos, actual_residue=target_pos % 8,
                               completion=completion,
                               parsed_answer=answer, correct=answer == case['gold'])
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    out.flush()
                    print(f'{protocol} {number}/{len(cases)} {case["case_id"]}: '
                          f'{int(row["correct"])} {completion!r}', flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_summary(rows, args.output / 'summary.json')


if __name__ == '__main__':
    main()
