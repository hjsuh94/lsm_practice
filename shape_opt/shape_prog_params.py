import os, time

import numpy as np
import torch
import matplotlib.pyplot as plt
import skfmm

class SMPParameters():
    def __init__(self):
        self.d = None # dimension of the program.
        self.n_grid = 200
        self.max_iters = 500
        self.gpu = True
        self.log_iterations = True
        self.cfl_integration = True
        
        self.h = 2e-2
        self.sigma = 1.0
        self.ksize_gaussian = 5
        
        self.rho_al_np = 100
        self.rho_al_V = 1
        
class SMPParameters2D(SMPParameters):
    def __init__(self):
        super().__init__()
        self.d = 2

class SMPParameters3D(SMPParameters):
    def __init__(self):
        super().__init__()
        self.d = 3
