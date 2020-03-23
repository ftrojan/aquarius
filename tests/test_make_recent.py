import datetime
import numpy as np
import pandas as pd
import utils


def test_make_recent_20200215():
    config = utils.get_config()
    ded = pd.datetime(2020, 2, 15)
    cal = utils.make_recent(ded, config)
    assert cal[ded] is np.True_
    assert cal[ded - datetime.timedelta(days=365*config['recent_time_window_years']-1)] is np.True_
    assert cal[ded - datetime.timedelta(days=365*config['recent_time_window_years'])] is np.False_
    assert len(cal) >= 365*(config['recent_time_window_years'] + config['preceding_time_window_max_years'])
