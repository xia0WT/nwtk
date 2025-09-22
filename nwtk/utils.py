import logging
import numpy as np
import random
import torch
import os

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

# early_stopping.py
#Copyright (c) 2018 Bjarte Mehus Sunde
#https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self,
		saver,  #     timm.utils.checkpoint_saver.CheckpointSaver
		trace_func, # logger
		patience=10,
		verbose=True,
		delta=0,
	):
	
		self.saver = saver
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_val_loss = None
		self.early_stop = False
		self.val_loss_min = np.inf
		self.delta = delta
		self.trace_func = trace_func
	
	def __call__(self, val_loss, epoch):
		self.epoch = epoch
	# Check if validation loss is nan
		if np.isnan(val_loss):
			self.trace_func("Validation loss is NaN. Ignoring this epoch.")
			return
	
		if self.best_val_loss is None:
			self.best_val_loss = val_loss
		elif val_loss < self.best_val_loss - self.delta:
	# Significant improvement detected
			self.best_val_loss = val_loss
			self.save_checkpoint(val_loss)
			self.counter = 0  # Reset counter since improvement occurred
		else:
	# No significant improvement
			self.counter += 1
			self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
	
	def save_checkpoint(self, val_loss):
		if self.verbose:
			self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
		save_path = os.path.join(self.saver.checkpoint_dir, f"checkpoint-earlystop.pth.tar")
		self.saver._save(save_path, self.epoch, metric = val_loss)
		self.val_loss_min = val_loss
