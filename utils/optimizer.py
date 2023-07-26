from torch_lr_finder import LRFinder
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR



class Optimization:
    def __init__(self, model, device, train_loader, criterion, num_epochs=24, scheduler=None, scheduler_args = []):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.optimizer = None
        self.define_optimizer()
        if scheduler:
            self.define_scheduler(scheduler, scheduler_args)
        else:
            self.scheduler = None
    
    def define_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.03, weight_decay=1e-4)


    def define_scheduler(self, scheduler, scheduler_args):
        if scheduler == 'OneCycleLR':
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr = scheduler_args['max_lr'], 
                steps_per_epoch = len(self.train_loader), 
                epochs = self.num_epochs, 
                pct_start = scheduler_args['pct_start'],
                div_factor = scheduler_args['div_factor'], 
                three_phase = scheduler_args['three_phase'], 
                final_div_factor = scheduler_args['final_div_factor'], 
                anneal_strategy = scheduler_args['anneal_strategy'], 
                verbose=True)
        elif scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode=scheduler_args['mode'], 
                factor=scheduler_args['factor'], 
                patience=scheduler_args['patience'], 
                threshold=scheduler_args['threshold'],
                verbose=True)
        else:
            self.scheduler = None

            

