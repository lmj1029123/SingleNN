import torch 
import numpy as np
from preprocess import train_val_split, train_test_split
from ContextManager import cd



Name = 'Si'
seed = 1

with cd(Name+f'-{seed}'):
    train_test_split(train_percent = 0.8, seed = seed)
    train_val_split(train_percent = 0.9, seed = 42)
