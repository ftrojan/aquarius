import numpy as np
import utils
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler()],
)


def test_ecdf():
    xdata = np.array([-1, 0, 2, 2, 1])
    cdf = utils.ECDF().fit(xdata)
    x = np.array([-3, 3, 0, 1, 1.5])
    cdf_expected = np.array([0, 1, 0.3, 0.5, 0.7])
    cdf_result = cdf.eval(x)
    logging.debug(cdf_result)
    xq_result = cdf.quantile(cdf_expected)
    xq_expected = np.array([-1, 2, 0, 1, 1.5])
    logging.debug(xq_result)
    assert np.all(cdf.x_values == np.array([-1, 0, 1, 2]))
    assert np.all(cdf.cdf_values == np.array([0.1, 0.3, 0.5, 0.9]))
    assert np.all(cdf_result == cdf_expected)
    assert np.all(xq_result == xq_expected)


def test_ecdf_weighted_equal():
    xdata = np.array([-1, 0, 2, 2, 1])
    wdata = np.array([1, 1, 1, 1, 1])
    cdf = utils.ECDF().fit(xdata, weights=wdata)
    x = np.array([-3, 3, 0, 1, 1.5])
    cdf_expected = np.array([0, 1, 0.3, 0.5, 0.7])
    cdf_result = cdf.eval(x)
    logging.debug(cdf_result)
    xq_result = cdf.quantile(cdf_expected)
    xq_expected = np.array([-1, 2, 0, 1, 1.5])
    logging.debug(xq_result)
    assert np.all(cdf.x_values == np.array([-1, 0, 1, 2]))
    assert np.all(cdf.cdf_values == np.array([0.1, 0.3, 0.5, 0.9]))
    assert np.all(cdf_result == cdf_expected)
    assert np.all(xq_result == xq_expected)


def test_ecdf_weighted_unequal():
    xdata = np.array([-1, 0, 2, 2, 1])
    wdata = np.array([2, 2, 3, 2, 1])
    cdf = utils.ECDF().fit(xdata, weights=wdata)
    x = np.array([-3, 3, 0, 1, 1.5])
    cdf_expected = np.array([0, 1, 0.3, 0.45, 0.675])
    cdf_result = cdf.eval(x)
    logging.debug(cdf_result)
    xq_result = cdf.quantile(cdf_expected)
    xq_expected = np.array([-1, 2, 0, 1, 1.5])
    logging.debug(xq_result)
    assert np.all(cdf.x_values == np.array([-1, 0, 1, 2]))
    assert np.all(cdf.cdf_values == np.array([0.1, 0.3, 0.45, 0.9]))
    assert np.all(cdf_result == cdf_expected)
    assert np.all(xq_result == xq_expected)
