import datetime, time
import os
import logging

def setup_logger(path, logger_name):

    fmt = "%(asctime)s.%(msecs).03d [%(filename)s:%(lineno)s - %(funcName)s()] %(levelname)s: %(threadName)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    # append a date string to the logger name so each file doesn't get too big
    datestr = datetime.datetime.utcnow().strftime('%Y%m%d')
    logname = path + logger_name + '_' + datestr + '.log'
    if not os.path.exists(path): os.mkdir(path)

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        datefmt=datefmt,
                        filename=logname,
                        filemode='a')

    # define a Handler which writes ERROR messages or higher to sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    formatter = logging.Formatter(fmt,datefmt=datefmt)
    formatter.converter = time.gmtime
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # return the logger
    logger = logging.getLogger(logger_name)
    return logger
    
    
