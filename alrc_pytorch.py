import torch

class ALRC(torch.nn.Module):
    """Adaptive learning rate clipping (ALRC) of outlier losses."""
    
    EPSILON = 1e-8  # Small constant for stability / prevent div 0
    
    def __init__(self, num_stddev=3, decay=0.999, initial_mean=25, initial_moment=900):
        super().__init__()
        self.num_stddev = num_stddev
        self.decay = decay
        # mu1
        self.mean = torch.tensor(initial_mean, dtype=torch.float)
        # mu2
        self.second_moment = torch.tensor(initial_moment, dtype=torch.float)

    def forward(self, loss):
        """Forward pass to clip the outlier losses."""
        
        # Calculate st dev
        std_dev = torch.sqrt(self.second_moment - self.mean**2 + self.EPSILON)
        
        # Clip losses
        threshold = self.mean + self.num_stddev * std_dev
        clipped_loss = torch.where(loss < threshold, loss, threshold)
        
        # Update mean and second moment using exponential moving average

        # Loss derived params so uses torch.no_grad()
        with torch.no_grad():
            mean_loss = torch.mean(clipped_loss)
            mean_loss_squared = torch.mean(clipped_loss ** 2)
            
            self.mean = self.decay * self.mean + (1 - self.decay) * mean_loss
            self.second_moment = self.decay * self.second_moment + (1 - self.decay) * mean_loss_squared
        
        return clipped_loss
