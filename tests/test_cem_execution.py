import numpy as np
import pytest

from render_cem_execution import decision_offsets


def test_decision_offsets_detect_replanning_boundaries():
    assert decision_offsets(np.array([0, 0, 0, 1, 1])) == [0, 3]


def test_decision_offsets_rejects_empty_input():
    with pytest.raises(ValueError):
        decision_offsets(np.array([], dtype=np.int64))
