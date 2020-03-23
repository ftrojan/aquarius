import numpy as np
import utils


def test_ecdf():
    xdata = np.array([-1, 0, 2, 2, 1])
    cdf = utils.ECDF().fit(xdata)
    x = np.array([-3, 3, 0, 1, 1.5])
    cdf_expected = np.array([0, 1, 0.4, 0.6, 0.8])
    cdf_result = cdf.eval(x)
    assert np.all(cdf.x_values == np.array([-1, 0, 1, 2]))
    assert np.all(cdf.cdf_values == np.array([0.2, 0.4, 0.6, 1.0]))
    assert np.all(cdf_result == cdf_expected)
