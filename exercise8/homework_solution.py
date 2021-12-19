import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 

class Oper(nn.Module):
    def __init__(self, func=None):
        super().__init__()
        self.func = func

    def forward(self, x):
        if self.func is not None:
            return self.func(x)
        return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResidualBlock(nn.Module):
    """
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        stride: Stride size of the first convolution, used for downsampling
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels != out_channels or stride > 1:
            self.skip = Oper(
                lambda x: F.pad(
                    x[:, :, ::stride, ::stride],
                    (0, 0, 0, 0, 0, out_channels - in_channels),
                    mode="constant",
                    value=0
                )
            )
        else:
            self.skip = Oper(func=None)

        self.path = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )

    def forward(self, input):
        return F.relu(self.path(input) + self.skip(input))


class ResidualStack(nn.Module):
    """
    A stack of resudual blocks.

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        stride: Stride size of the first layer, used for downsampling
        num_blocks: Number of residual blocks
    """

    # iportant: We cannot use python list here. We must use nn.ModuleList, otherwise
    # the modules cannot be transfered to GPU.
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        block_list = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            block_list.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1))
        self.block_list = nn.ModuleList(block_list)

    def forward(self, input):
        x = input
        for block in self.block_list:
            x = block(x)
        return x


n = 5
num_classes = 10
resnet = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    ResidualStack(16, 16, stride=1, num_blocks=n),
    ResidualStack(16, 32, stride=2, num_blocks=n),
    ResidualStack(32, 64, stride=2, num_blocks=n),
    nn.AdaptiveAvgPool2d(output_size=1),
    Oper(lambda x: x.squeeze()),
    nn.Linear(64, num_classes)
)

#=========================================================Solution



#  class ResidualBlock(nn.Module):
#      """
#      The residual block used by ResNet.
#
#      Args:
#          in_channels: The number of channels (feature maps) of the incoming embedding
#          out_channels: The number of channels after the first convolution
#          stride: Stride size of the first convolution, used for downsampling
#      """
#
#      def __init__(self, in_channels, out_channels, stride=1):
#          super().__init__()
#          if stride > 1 or in_channels != out_channels:
#              # Add strides in the skip connection and zeros for the new channels.
#              self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],
#                                                 (0, 0, 0, 0, 0, out_channels - in_channels),
#                                                 mode="constant", value=0))
#          else:
#              self.skip = nn.Sequential()
#
#          # TODO: Initialize the required layers
#          self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
#          self.bn1 = nn.BatchNorm2d(out_channels)
#          self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)
#          self.bn2 = nn.BatchNorm2d(out_channels)
#
#      def forward(self, input):
#          # TODO: Execute the required layers and functions
#          x1 = F.relu(self.bn1(self.conv1(input)))
#          x2 = self.bn2(self.conv2(x1))
#          return F.relu(x2 + self.skip(input))


#  class ResidualStack(nn.Module):
#      """
#      A stack of residual blocks.
#
#      Args:
#          in_channels: The number of channels (feature maps) of the incoming embedding
#          out_channels: The number of channels after the first layer
#          stride: Stride size of the first layer, used for downsampling
#          num_blocks: Number of residual blocks
#      """
#
#      def __init__(self, in_channels, out_channels, stride, num_blocks):
#          super().__init__()
#
#          # TODO: Initialize the required layers (blocks)
#          blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
#          for _ in range(num_blocks - 1):
#              blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
#          self.blocks = nn.ModuleList(blocks)
#
#      def forward(self, input):
#          # TODO: Execute the layers (blocks)
#          x = input
#          for block in self.blocks:
#              x = block(x)
#          return x

#  n = 5
#  num_classes = 10
#
#  # TODO: Implement ResNet via nn.Sequential
#  resnet = nn.Sequential(
#      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
#      nn.BatchNorm2d(16),
#      nn.ReLU(),
#      ResidualStack(16, 16, stride=1, num_blocks=n),
#      ResidualStack(16, 32, stride=2, num_blocks=n),
#      ResidualStack(32, 64, stride=2, num_blocks=n),
#      nn.AdaptiveAvgPool2d(1),
#      Lambda(lambda x: x.squeeze()),
#      nn.Linear(64, num_classes)
#  )


def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        
resnet.apply(initialize_weight);



class CIFAR10Subset(torchvision.datasets.CIFAR10):
    """
    Get a subset of the CIFAR10 dataset, according to the passed indices.
    """
    def __init__(self, *args, idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if idx is None:
            return
        
        self.data = self.data[idx]
        targets_np = np.array(self.targets)
        self.targets = targets_np[idx].tolist()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
])
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

ntrain = 45_000
train_set = CIFAR10Subset(root='./data', train=True, idx=range(ntrain),
                          download=True, transform=transform_train)
val_set = CIFAR10Subset(root='./data', train=True, idx=range(ntrain, 50_000),
                        download=True, transform=transform_eval)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_eval)

                                        
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=128,
                                                   shuffle=True, num_workers=2,
                                                   pin_memory=True)
dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=128,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=True)
dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=128,
                                                  shuffle=False, num_workers=2,
                                                  pin_memory=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet.to(device);


def run_epoch(model, optimizer, dataloader, train):
    """
    Run one epoch of training or evaluation.
    
    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        dataloader: Dataloader providing the data to run our model on
        train: Whether this epoch is used for training or evaluation
        
    Returns:
        Loss and accuracy in this epoch.
    """
    # TODO: Change the necessary parts to work correctly during evaluation (train=False)
    
    device = next(model.parameters()).device
    
    # Set model to training mode (for e.g. batch normalization, dropout)
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0

    # Iterate over data
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = F.cross_entropy(pred, yb)
            top1 = torch.argmax(pred, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        # statistics
        epoch_loss += loss.item()
        epoch_acc += ncorrect.item()
    
    epoch_loss /= len(dataloader.dataset)
    epoch_acc /= len(dataloader.dataset)
    return epoch_loss, epoch_acc

def fit(model, optimizer, lr_scheduler, dataloaders, max_epochs, patience):
    """
    Fit the given model on the dataset.
    
    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        lr_scheduler: Learning rate scheduler that improves training
                      in late epochs with learning rate decay
        dataloaders: Dataloaders for training and validation
        max_epochs: Maximum number of epochs for training
        patience: Number of epochs to wait with early stopping the
                  training if validation loss has decreased
                  
    Returns:
        Loss and accuracy in this epoch.
    """
    
    best_acc = 0
    curr_patience = 0
    
    for epoch in range(max_epochs):
        train_loss, train_acc = run_epoch(model, optimizer, dataloaders['train'], train=True)
        lr_scheduler.step()
        print(f"Epoch {epoch + 1: >3}/{max_epochs}, train loss: {train_loss:.2e}, accuracy: {train_acc * 100:.2f}%")
        
        val_loss, val_acc = run_epoch(model, None, dataloaders['val'], train=False)
        print(f"Epoch {epoch + 1: >3}/{max_epochs}, val loss: {val_loss:.2e}, accuracy: {val_acc * 100:.2f}%")
        
        # TODO: Add early stopping and save the best weights (in best_model_weights)
        if val_acc >= best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
                
        # Early stopping
        if epoch - best_epoch >= patience:
            break
    
    model.load_state_dict(best_model_weights)


optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# Fit model
fit(resnet, optimizer, lr_scheduler, dataloaders, max_epochs=200, patience=50)
