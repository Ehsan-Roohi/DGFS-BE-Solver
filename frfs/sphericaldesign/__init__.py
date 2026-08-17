# -*- coding: utf-8 -*-

from importlib.resources import files
import re

import numpy as np


class BaseTabulatedSphericalQuadRule(object):
    def __init__(self, rule):
        pts = []
        wts = []

        rule = re.sub(r'(?<=\))\s*,?\s*(?!$)', r'\n', rule)
        rule = re.sub(r'\(|\)|,', '', rule).strip()
        rule = rule[1:-1] if rule.startswith('[') else rule

        for l in rule.splitlines():
            if not l:
                continue

            # Parse the line
            args = [float(f) for f in l.split()]

            if len(args) == self.ndim:
                pts.append(args)
            elif len(args) == self.ndim + 1:
                pts.append(args[:-1])
                wts.append(args[-1])
            else:
                raise ValueError('Invalid points in quadrature rule')

        if len(wts) and len(wts) != len(pts):
            raise ValueError('Invalid number of weights')

        # Flatten 1D rules
        if self.ndim == 1:
            pts = [p[0] for p in pts]

        # Cast
        self.pts = np.array(pts, dtype=float)
        self.wts = np.array(wts, dtype=float)


class BaseStoredSphericalQuadRule(BaseTabulatedSphericalQuadRule):
    @classmethod
    def _iter_rules(cls):
        rpaths = getattr(cls, '_rpaths', None)
        if rpaths is None:
            root = files(__name__).joinpath(cls.shape)
            cls._rpaths = rpaths = [item.name for item in root.iterdir()]

        for path in rpaths:
            m = re.match(r'([a-zA-Z0-9\-~+]+)-ss(\d+)'
                         r'(?:-m(\d+))?\.txt$', path)
            if m:
                yield (path, m.group(1), int(m.group(2)),
                       int(m.group(3) or -1))

    def __init__(self, name=None, npts=None, qdeg=None, flags=None):
        if not npts and not qdeg:
            raise ValueError('Must specify either npts or qdeg')

        best = None
        for rpath, rname, rqdeg, rnpts in self._iter_rules():
            # See if this rule fulfils the required criterion
            if ((not name or name == rname) and
                (not npts or npts == rnpts) and
                (not qdeg or qdeg <= rqdeg)):
                # If so see if it is better than the current candidate
                if (not best or
                    (npts and rqdeg > best[2]) or
                    (qdeg and rnpts < best[1])):
                    best = (rpath, rnpts, rqdeg)

        # Raise if no suitable rules were found
        if not best:
            raise ValueError('No suitable spherical quadrature rule found')

        # Load the rule
        rule = files(__name__).joinpath(self.shape, best[0]).read_text(
            encoding='utf-8'
        )
        super().__init__(rule)


def get_sphquadrule(sstype, rule=None, npts=None, qdeg=None, flags=None):
    ndims = dict(symmetric=3)

    if rule and not re.match(r'[a-zA-z0-9\-~+]+$', rule):
        class TabulatedSphericalQuadRule(BaseTabulatedSphericalQuadRule):
            shape = sstype
            ndim = ndims[sstype]

        r = TabulatedSphericalQuadRule(rule)

        # Validate the provided point set
        if npts and npts != len(r.pts):
            raise ValueError('Invalid number of points in provided rule')

        if qdeg and not len(r.wts):
            raise ValueError('Provided rule has no quadrature weights')

        return r
    else:
        class StoredSphericalQuadRule(BaseStoredSphericalQuadRule):
            shape = sstype
            ndim = ndims[sstype]

        return StoredSphericalQuadRule(rule, npts, qdeg, flags)
