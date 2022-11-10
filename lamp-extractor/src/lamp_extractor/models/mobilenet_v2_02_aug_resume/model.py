from torch import nn
from torchvision import models
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_points = 4
        self.num_features = 2
        self.model=models.mobilenet_v2(pretrained=False)
        self.in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Identity()
        self.model_name="mobilenet_v2"
        self.model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.headL = nn.Linear(self.in_features, self.num_points * self.num_features)
        
    def forward(self, x):
        """
        pred = [
             [tlx, tly],
             [trx, try],
             [brx, bry],
             [blx, bly]
        ]
        """
        x = self.model(x)
        x = self.headL(x)
        #x = torch.tanh(x)
       
        return x.view((-1, self.num_points, self.num_features))

if __name__ == "__main__":
    net = Net()
    input = torch.ones([1, 3, 64, 64], dtype=torch.float32) * 100
    pred = net.forward(input)
    print(pred)
    print(pred.max(), pred.min())
