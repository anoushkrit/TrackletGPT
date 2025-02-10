# project specific librares
from tools import builder

class RemoteDataLoader:
    def __init__(self):
        self.sampler = None
        self.dataloader = None
        self.iter_loader = None

    def initialize_dataset(self, args, config):

        config._base_.num_workers = args.num_workers
        self.sampler, self.dataloader = builder.dataset_builder(args, config)
        self.iter_loader = iter(self.dataloader)
    
    def get_batch(self):
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)
    
    def get_batch_size(self):
        return len(self.dataloader)
