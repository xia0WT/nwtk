import logging
import numpy as np
import random
import torch

import logging

class create_log:
    def __init__(self, file):
        
        self.file = file
        
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(fmt)
        
        self.fh = logging.FileHandler(self.file)
        self.fh.setFormatter(fmt)
        
    def __call__(self, name="Admin"):
        
        logging.getLogger().setLevel(logging.INFO)
        logger = logging.getLogger(name)
        logger.handlers.clear()
   
        logger.addHandler(self.ch)
        logger.addHandler(self.fh)

        return logger


def seed_everything(seed):
	random.seed(seed)
	# os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)         # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True