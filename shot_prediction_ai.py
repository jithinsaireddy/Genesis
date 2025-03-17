import torch.nn as nn

class ShotPredictionAI(nn.Module):
    def __init__(self):
        super(ShotPredictionAI, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256 * 256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # Predicts one of 5 shot types
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
shot_ai = ShotPredictionAI()
