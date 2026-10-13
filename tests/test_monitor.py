from revenueiq.monitor import population_stability_index


def test_psi_identical():
    assert abs(population_stability_index(list(range(100)), list(range(100)))) < 1e-9


def test_psi_shift():
    assert population_stability_index(list(range(100)), list(range(100, 200))) > 0.1
