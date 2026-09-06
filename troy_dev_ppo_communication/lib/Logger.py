import logging 

class Logger:
    def __init__(self, name, level=20): # 10 for debug, 20 for info, 30 for warning, 40 for error, 50 for critical 
        self.logger= logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")# print time 
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    def print(self, message, level=20):
        self.logger.log(level, message)
