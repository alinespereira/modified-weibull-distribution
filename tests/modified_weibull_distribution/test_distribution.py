from hypothesis import given, strategies as st, settings
import pytest
from scipy import stats  # type: ignore

from modified_weibull_distribution import mwd


@pytest.fixture(scope="session")
def significance_level() -> float:
    return 0.05


@pytest.fixture(scope="session")
def replications() -> int:
    return 10


@pytest.fixture(scope="session")
def tolerance() -> float:
    return 2 / 10


@given(
    alpha=st.floats(min_value=0).filter(lambda x: x > 0),
    beta=st.floats(min_value=0).filter(lambda x: x > 0),
    gamma=st.floats(min_value=0).filter(lambda x: x > 0),
    sample_size=st.integers(min_value=1),
)
@settings(max_examples=10)
def test_modified_weibull_kolmogorov_smirnoff(
    alpha: float,
    beta: float,
    gamma: float,
    sample_size: int,
    significance_level: float,
    replications: int,
    tolerance: float,
):
    random_variable = mwd(alpha, beta, gamma)

    rejected: int = 0
    samples = random_variable.rvs(size=(replications, sample_size))
    for sample in samples:
        ks_result = stats.kstest(sample, random_variable.cdf)
        rejected += ks_result.pvalue < significance_level
    assert rejected / replications < tolerance
