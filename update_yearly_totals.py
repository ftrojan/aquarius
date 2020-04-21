import logging
import utils
import constants


logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler(), logging.FileHandler('../../data/logs/update_yearly_totals.log', 'a')],
)
logging.debug("started")
current_year = utils.get_current_year()
utils.update_yearly_totals(current_year)
logging.debug("completed")
