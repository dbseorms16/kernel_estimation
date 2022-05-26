import logging
logger = logging.getLogger('base')
from .B_model import B_Model as M


def create_model(opt):
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
