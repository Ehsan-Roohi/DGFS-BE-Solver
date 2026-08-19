#!/usr/bin/env python3
"""CPU smoke tests for the phase-3 conservative projection reference."""
from __future__ import annotations

import unittest

import numpy as np

from conservative_projection import ConservativeProjector


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        grid = np.linspace(-2.0, 2.0, 8)
        cls.cv = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1)
        cls.P = ConservativeProjector(cls.cv, 0.125)
        rng = np.random.default_rng(20260819)
        cls.f = np.exp(-np.sum(cls.cv * cls.cv, axis=0))
        cls.q = rng.normal(size=cls.f.size)

    def test_euclidean_cancels_five_moments(self):
        result = self.P.project(self.q, self.f, "euclidean")
        scale = np.maximum(self.P.cancellation_scales(result.Qc), 1.0)
        self.assertLess(np.max(np.abs(result.moments_after) / scale), 2.0e-14)

    def test_fplus_cancels_five_moments(self):
        result = self.P.project(self.q, self.f, "fplus")
        scale = np.maximum(self.P.cancellation_scales(result.Qc), 1.0)
        self.assertLess(np.max(np.abs(result.moments_after) / scale), 2.0e-14)
        self.assertGreaterEqual(result.weight_min, 0.0)

    def test_projecting_conservative_term_is_identity(self):
        conservative = self.P.project(self.q, self.f, "euclidean").Qc
        twice = self.P.project(conservative, self.f, "euclidean").Qc
        np.testing.assert_allclose(twice, conservative, rtol=0.0, atol=2.0e-13)


if __name__ == "__main__":
    unittest.main()
