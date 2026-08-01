import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_cem_population import cost_normalizer, validate_trace


def synthetic_trace():
    return {
        "costs": np.arange(2 * 1 * 4, dtype=np.float32).reshape(2, 1, 4),
        "elite_indices": np.array([[[0, 1]], [[1, 2]]], dtype=np.int64),
        "predicted_emb": np.ones((2, 1, 4, 6, 192), dtype=np.float32),
    }


def test_population_trace_shape_and_cost_scale_are_valid():
    trace = synthetic_trace()
    assert validate_trace(trace, environment=0) == (2, 4, 2)
    scale = cost_normalizer(trace["costs"])
    assert scale.vmin < scale.vmax


def test_population_trace_rejects_invalid_elite_index():
    trace = synthetic_trace()
    trace["elite_indices"][0, 0, 0] = 4
    with pytest.raises(ValueError, match="outside"):
        validate_trace(trace, environment=0)
