#!/usr/bin/env python3
"""Apply the phase-3 conservative-projection hook to a DGFS-BE-Solver checkout.

    python apply_hook.py /path/to/DGFS-BE-Solver

It (1) copies ``projection.py`` to ``frfs/solvers/dgfs/projection.py`` and
(2) patches ``frfs/solvers/dgfs/system.py`` so that

    [scattering-model]
    projection = none | euclidean | f | fplus      (default none)
    projection-solve = device | host               (default device)

controls an in-place five-moment conservative projection of Q(f,f) applied right
after every ``self.sm.fs(...)`` call inside ``DGFSSystem.collide``.  The patch is
idempotent and only touches two anchors; with ``projection = none`` the solver
is bit-identical to the unpatched code path.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HOOK_INIT = '''        self.sm = scatteringcls(backend, self.cfg, self.vm)

        # --- phase-3: optional five-moment conservative projection of Q ---
        _proj = str(cfg.get('scattering-model', 'projection', 'none')).strip().lower()
        self.projector = None
        if _proj not in ('none', 'off', 'false', '0', ''):
            from frfs.solvers.dgfs.projection import GPUConservativeProjector
            _solve = str(cfg.get('scattering-model', 'projection-solve', 'device')).strip().lower()
            self.projector = GPUConservativeProjector(backend, self.vm, _proj, _solve)
            print("Conservative projection of Q: weighting=%s solve=%s" % (_proj, _solve))
        # ------------------------------------------------------------------
'''
ANCHOR_INIT = "        self.sm = scatteringcls(backend, self.cfg, self.vm)\n"
ANCHOR_COLLIDE = "                self.sm.fs(arr_in, arr_out, elem, upt)\n"
HOOK_COLLIDE = ANCHOR_COLLIDE + '''                if self.projector is not None:
                    self.projector.apply(arr_in, arr_out, elem, upt)
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).resolve()
    dgfs = root / "frfs" / "solvers" / "dgfs"
    system_py = dgfs / "system.py"
    if not system_py.is_file():
        raise SystemExit(f"not a DGFS-BE-Solver checkout: {system_py} missing")
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "projection.py", dgfs / "projection.py")
    src = system_py.read_text()
    if "self.projector" in src:
        print("system.py already patched")
    else:
        if src.count(ANCHOR_INIT) != 1 or src.count(ANCHOR_COLLIDE) != 1:
            raise SystemExit("anchors not found exactly once in system.py; refusing an ambiguous patch")
        src = src.replace(ANCHOR_INIT, HOOK_INIT).replace(ANCHOR_COLLIDE, HOOK_COLLIDE)
        system_py.write_text(src)
        print(f"patched {system_py}")
    print(f"installed {dgfs / 'projection.py'}")


if __name__ == "__main__":
    main()
