#!/usr/bin/env python3
"""With the mixed split off, the engine sources must be vanilla's, line for line.

Resolves every `#if defined(STRATA_MIXED_SPLIT)` block as false in each file this branch changes outside `mixed_split/`
and compares the result with the same file at the vanilla reference (default: c775728, vanilla 0.2.15).  Only the
form `#if defined(STRATA_MIXED_SPLIT)` with an optional `#else` is handled; anything else is refused.  Needs no GPU
and no build.

  python3 mixed_split/tools/check_off_is_vanilla.py [REF]
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MACRO = 'STRATA_MIXED_SPLIT'


def resolve_off(macro, path):
    """FILE as it reads with `#if defined(MACRO)` false (lines kept, in order)."""
    ours = re.compile(r'^\s*#\s*if\s+defined\(\s*' + re.escape(macro) + r'\s*\)\s*(//.*)?$')
    any_if = re.compile(r'^\s*#\s*if(n?def)?\b')
    els = re.compile(r'^\s*#\s*else\b')
    elif_ = re.compile(r'^\s*#\s*elif\b')
    endif = re.compile(r'^\s*#\s*endif\b')
    stack = []  # entries: 'ours-if' (dropping), 'ours-else' (keeping), 'other'
    out = []
    nums = []
    for n, line in enumerate(open(path), 1):
        if macro in line and line.lstrip().startswith('#') and not ours.match(line) and not endif.match(line) \
                and not els.match(line) and 'define' not in line.split(macro)[0][-8:]:
            raise SystemExit(f'{path}:{n}: unhandled form: {line.strip()}')
        dropping = any(s == 'ours-if' for s in stack)
        if ours.match(line):
            stack.append('ours-if')
            continue
        if any_if.match(line):
            stack.append('other')
            if not dropping:
                (out.append(line), nums.append(n))
            continue
        if elif_.match(line):
            if stack and stack[-1].startswith('ours'):
                raise SystemExit(f'{path}:{n}: #elif in a {macro} block')
            if not dropping:
                (out.append(line), nums.append(n))
            continue
        if els.match(line):
            if stack and stack[-1] == 'ours-if':
                stack[-1] = 'ours-else'
                continue
            if not dropping:
                (out.append(line), nums.append(n))
            continue
        if endif.match(line):
            top = stack.pop()
            if top == 'other' and not dropping:
                (out.append(line), nums.append(n))
            continue
        if not dropping:
            (out.append(line), nums.append(n))
    if stack:
        raise SystemExit(f'{path}: unbalanced conditionals')
    return ''.join(out)


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else 'c775728'
    names = subprocess.run(['git', '-C', str(REPO), 'diff', '--name-only', ref, '--', 'src', 'include'],
                           text=True, capture_output=True, check=True).stdout.split()
    bad = 0
    for name in names:
        path = REPO / name
        if not path.is_file():
            print(f'removed from vanilla: {name}')
            bad += 1
            continue
        vanilla = subprocess.run(['git', '-C', str(REPO), 'show', f'{ref}:{name}'], text=True, capture_output=True)
        if vanilla.returncode != 0:
            print(f'not in vanilla (a new file outside mixed_split/): {name}')
            bad += 1
            continue
        same = resolve_off(MACRO, path) == vanilla.stdout
        print(f'{"same" if same else "DIFFERS"}: {name}')
        bad += not same
    print(f'{len(names) - bad} of {len(names)} changed files are vanilla with {MACRO} off')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
