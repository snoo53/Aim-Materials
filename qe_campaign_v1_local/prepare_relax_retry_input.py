#!/usr/bin/env python3
"""
Build a rescue QE relax input from a previous non-converged run.

Strategy:
- keep original input as baseline;
- if available, replace geometry with the last ATOMIC_POSITIONS / CELL_PARAMETERS
  found in qe_relax.out;
- increase ionic/electronic iteration budget and soften mixing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple


def parse_nat(text: str) -> Optional[int]:
    m = re.search(r"(?im)^\s*nat\s*=\s*([0-9]+)\s*,?\s*$", text)
    if not m:
        return None
    return int(m.group(1))


def _is_atom_line(s: str) -> bool:
    parts = s.split()
    if len(parts) < 4:
        return False
    return re.match(r"^[A-Za-z][A-Za-z0-9_]*$", parts[0]) is not None


def parse_last_atomic_positions(
    lines: List[str], nat: Optional[int]
) -> Tuple[Optional[str], Optional[List[str]]]:
    last_header = None
    last_block = None
    for i, line in enumerate(lines):
        if not line.strip().startswith("ATOMIC_POSITIONS"):
            continue
        header = line.rstrip("\n")
        block: List[str] = []
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                if block:
                    break
                j += 1
                continue
            if s.startswith("CELL_PARAMETERS") or s.startswith("K_POINTS"):
                break
            if s.startswith("&") or s.startswith("/"):
                break
            if not _is_atom_line(s):
                break
            block.append(lines[j].rstrip("\n"))
            j += 1
            if nat is not None and len(block) >= nat:
                break
        if nat is not None and len(block) != nat:
            continue
        if not block:
            continue
        last_header = header
        last_block = block
    return last_header, last_block


def parse_last_cell_parameters(lines: List[str]) -> Tuple[Optional[str], Optional[List[str]]]:
    last_header = None
    last_block = None
    for i, line in enumerate(lines):
        if not line.strip().startswith("CELL_PARAMETERS"):
            continue
        header = line.rstrip("\n")
        block: List[str] = []
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                if block:
                    break
                j += 1
                continue
            if s.startswith("ATOMIC_POSITIONS") or s.startswith("K_POINTS"):
                break
            if s.startswith("&") or s.startswith("/"):
                break
            parts = s.split()
            if len(parts) < 3:
                break
            block.append(lines[j].rstrip("\n"))
            j += 1
            if len(block) >= 3:
                break
        if len(block) == 3:
            last_header = header
            last_block = block
    return last_header, last_block


def find_block_range(lines: List[str], start_prefix: str) -> Tuple[Optional[int], Optional[int]]:
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(start_prefix):
            start = i
            break
    if start is None:
        return None, None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("ATOMIC_POSITIONS") and start_prefix != "ATOMIC_POSITIONS":
            end = j
            break
        if s.startswith("CELL_PARAMETERS") and start_prefix != "CELL_PARAMETERS":
            end = j
            break
        if s.startswith("K_POINTS"):
            end = j
            break
        if s.startswith("&"):
            end = j
            break
    return start, end


def replace_atomic_positions(lines: List[str], header: str, block: List[str]) -> List[str]:
    s, e = find_block_range(lines, "ATOMIC_POSITIONS")
    if s is None:
        return lines
    repl = [header + "\n"] + [x.rstrip("\n") + "\n" for x in block]
    return lines[:s] + repl + lines[e:]


def replace_cell_parameters(lines: List[str], header: str, block: List[str]) -> List[str]:
    s, e = find_block_range(lines, "CELL_PARAMETERS")
    if s is None:
        return lines
    repl = [header + "\n"] + [x.rstrip("\n") + "\n" for x in block]
    return lines[:s] + repl + lines[e:]


def find_namelist_range(lines: List[str], name: str) -> Tuple[Optional[int], Optional[int]]:
    start = None
    end = None
    target = "&" + name.upper()
    for i, line in enumerate(lines):
        if line.strip().upper() == target:
            start = i
            break
    if start is None:
        return None, None
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "/":
            end = j
            break
    return start, end


def remove_param_in_namelist(lines: List[str], name: str, key: str) -> List[str]:
    s, e = find_namelist_range(lines, name)
    if s is None or e is None:
        return lines
    out = lines[: s + 1]
    pat = re.compile(r"^\s*" + re.escape(key) + r"\s*=", re.IGNORECASE)
    for i in range(s + 1, e):
        if pat.search(lines[i]):
            continue
        out.append(lines[i])
    out.extend(lines[e:])
    return out


def upsert_param(lines: List[str], namelist: str, key: str, value: str) -> List[str]:
    lines = remove_param_in_namelist(lines, namelist, key)
    s, e = find_namelist_range(lines, namelist)
    if s is None or e is None:
        return lines
    insert_line = f"  {key} = {value},\n"
    return lines[:e] + [insert_line] + lines[e:]


def get_control_nstep(lines: List[str]) -> Optional[int]:
    s, e = find_namelist_range(lines, "CONTROL")
    if s is None or e is None:
        return None
    pat = re.compile(r"^\s*nstep\s*=\s*([0-9]+)\s*,?\s*$", re.IGNORECASE)
    for i in range(s + 1, e):
        m = pat.search(lines[i].strip())
        if m:
            return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--outlog", required=True)
    ap.add_argument("--nstep_add", type=int, default=180)
    ap.add_argument("--electron_maxstep", type=int, default=300)
    ap.add_argument("--mixing_beta", type=float, default=0.20)
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    log_path = Path(args.outlog)

    in_text = in_path.read_text(encoding="utf-8", errors="ignore")
    in_lines = in_text.splitlines(keepends=True)
    nat = parse_nat(in_text)

    out_lines = []
    if log_path.exists():
        out_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    src = "input"
    at_header, at_block = parse_last_atomic_positions(out_lines, nat) if out_lines else (None, None)
    cell_header, cell_block = parse_last_cell_parameters(out_lines) if out_lines else (None, None)

    if at_header and at_block:
        in_lines = replace_atomic_positions(in_lines, at_header, at_block)
        src = "output:last_atomic_positions"
    if cell_header and cell_block:
        in_lines = replace_cell_parameters(in_lines, cell_header, cell_block)
        if src.startswith("output:"):
            src += "+cell"

    prev_nstep = get_control_nstep(in_lines)
    new_nstep = (prev_nstep or 120) + int(args.nstep_add)

    in_lines = upsert_param(in_lines, "CONTROL", "nstep", str(new_nstep))
    in_lines = upsert_param(in_lines, "ELECTRONS", "electron_maxstep", str(int(args.electron_maxstep)))
    in_lines = upsert_param(in_lines, "ELECTRONS", "mixing_beta", f"{float(args.mixing_beta):.3f}")

    out_path.write_text("".join(in_lines), encoding="utf-8")
    print(
        f"[prepare_relax_retry_input] wrote={out_path} "
        f"geom_source={src} nat={nat} nstep={new_nstep} "
        f"electron_maxstep={int(args.electron_maxstep)} mixing_beta={float(args.mixing_beta):.3f}"
    )


if __name__ == "__main__":
    main()

