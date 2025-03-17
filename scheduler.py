import torch
import math
import torch.optim.lr_scheduler as lr_scheduler

class ExponentialLR:
    def __init__(self, optimizer, gamma=0.1, last_epoch=-1):
        self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    
    def get_scheduler(self):
        return self.scheduler

class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    def get_scheduler(self):
        return self.scheduler

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
    def get_scheduler(self):
        return self.scheduler


class OneCycleLR:
    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, last_epoch=-1):
        self.scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=None, epochs=epochs, steps_per_epoch=steps_per_epoch, last_epoch=last_epoch)
    def get_scheduler(self):
        return self.scheduler


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='max'):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode)
    def get_scheduler(self):
        return self.scheduler

def cosine_scheduler(optimizer, num_epochs, initial_lr=0.01, min_lr=0.0001):
    lr_lambda = lambda epoch: min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / num_epochs))
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



def instance_0(optimizer, steps, epochs):

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=steps, epochs=epochs, div_factor = 0.1, final_div_factor = 1)

    return scheduler

def instance_1(optimizer):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.5, min_lr=1e-4)

    return scheduler

def instance_2(optimizer, steps, epochs):
     
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=steps, anneal_strategy='linear', div_factor = 0.1, final_div_factor = 1, epochs=epochs)
    
    return scheduler


def instance_3(optimizer, steps, epochs): 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.5, min_lr=1e-4)

    return scheduler


