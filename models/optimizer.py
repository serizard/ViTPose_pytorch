import torch.optim as optim

# class LayerDecayOptimizer:
#     def __init__(self, optimizer, layerwise_decay_rate):
#         self.optimizer = optimizer
#         self.layerwise_decay_rate = layerwise_decay_rate
#         self.param_groups = optimizer.param_groups
        
#     def step(self, *args, **kwargs):
#         for i, group in enumerate(self.optimizer.param_groups):
#             group['lr'] *= self.layerwise_decay_rate[i]
#         self.optimizer.step(*args, **kwargs)
        
#     def zero_grad(self, *args, **kwargs):
#         self.optimizer.zero_grad(*args, **kwargs)

class LayerDecayOptimizer:
    def __init__(self, optimizer, layerwise_decay_rate, warmup_scheduler):
        self.optimizer = optimizer
        self.layerwise_decay_rate = layerwise_decay_rate
        self.warmup_scheduler = warmup_scheduler
        self.param_groups = optimizer.param_groups

    def step(self, *args, **kwargs):
        # Apply warmup_scheduler step to adjust learning rate
        self.warmup_scheduler.step(*args, **kwargs)

        # Apply layerwise decay after warmup_scheduler step
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] *= self.layerwise_decay_rate[i]

        # Perform the optimizer step
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)