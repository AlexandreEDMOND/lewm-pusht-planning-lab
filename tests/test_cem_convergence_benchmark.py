import numpy as np
import pytest

from cem_convergence_benchmark import convergence_summary, timing_for_environment


def test_convergence_uses_running_best_and_counts_candidates():
    costs = np.array([[10.0, 12.0], [7.0, 8.0], [4.0, 6.0], [3.0, 5.0]])
    summary = convergence_summary(costs, np.array([0.2, 0.3, 0.4, 0.5]))

    # 95% of the observed 10 -> 3 improvement means a cost <= 3.35 at iteration 4.
    assert summary["convergence_iteration"] == 4
    assert summary["candidate_trajectories_to_convergence"] == 8
    assert summary["candidate_trajectories_total"] == 8
    assert summary["compute_seconds_to_convergence"] == pytest.approx(1.4)


def test_timing_requires_a_single_environment_batch():
    metadata = {
        "environment_batch_slices": [[0, 1], [1, 2]],
        "cem_iteration_seconds_per_batch": [[0.1, 0.2], [0.3, 0.4]],
    }
    np.testing.assert_allclose(timing_for_environment(metadata, 1, 2), [0.3, 0.4])
    with pytest.raises(ValueError, match="batch_size=1"):
        timing_for_environment(
            {
                "environment_batch_slices": [[0, 2]],
                "cem_iteration_seconds_per_batch": [[0.1, 0.2]],
            },
            0,
            2,
        )
