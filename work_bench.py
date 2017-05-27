import logging
import zipfile

def init_my_logger(): 
    logger = logging.getLogger()  
    logger.setLevel(logging.DEBUG)
    
    logfile = './log.txt'  
    fh = logging.FileHandler(logfile, mode='w')  
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")  
    fh.setFormatter(formatter)  
    ch.setFormatter(formatter)  
    
    logger.addHandler(fh)  
    logger.addHandler(ch)  

    return logger

def test_log_module():
    LOG.info('this is a loggging info message')  
    LOG.debug('this is a loggging debug message')  
    LOG.warning('this is loggging a warning message')  
    LOG.error('this is an loggging error message')  
    LOG.critical('this is a loggging critical message') 

if __name__ == '__main__':
    LOG = init_my_logger()
    test_log_module()
    str = 'My name is %s. My age is %d' % ('xxx', 25)
    LOG.info(str)
    with zipfile.ZipFile("log.zip", "w") as fout:
        fout.write("log.txt", compress_type=zipfile.ZIP_DEFLATED)