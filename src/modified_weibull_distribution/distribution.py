import numpy as np
from scipy import stats # type: ignore
from scipy.stats import rv_continuous # type: ignore

class mwd_gen(rv_continuous):
    "Modified Weibull Distribution (MWD)"
    def _cdf(self, x, alpha, beta, gamma):
        return 1 - np.exp(-alpha * x - beta * np.power(x, gamma))

    def _pdf(self, x, alpha, beta, gamma):
        return (
            (alpha * x + beta * gamma * np.power(x, gamma - 1))
            * np.exp(-alpha * x - beta * np.power(x, gamma))
        )

    def _rvs(self, alpha, beta, gamma, size=None, random_state=None):
        u1 = stats.uniform.rvs(size=size, random_state=random_state)
        u2 = stats.uniform.rvs(size=size, random_state=random_state)
        x1 = -(1 / alpha) * np.log(1 - u1)
        x2 = (-(1 / beta) * np.log(1 - u2)) ** (1 / gamma)
        return np.minimum(x1, x2)

    def _logpdf(self, x, alpha, beta, gamma):
        return (
            np.log((alpha * x + beta * gamma * np.power(x, gamma - 1)))
            - alpha * x - beta * np.power(x, gamma)
        )

    def _logcdf(self, x, alpha, beta, gamma):
        return np.log(1 - np.exp(-alpha * x - beta * np.power(x, gamma)))

mwd = mwd_gen(a=0.0, name='mwd')
