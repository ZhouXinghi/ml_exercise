import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy 

# ==================================== 0. Dataloader

class CIFAR10Set(torchvision.datasets.CIFAR10):
    """
    Get a subset of CIFAR10 dataset, according to the passed indices
    """
    def __init__(self, root, train, transform, download, idx=None):
        super().__init__(root=root, train=train, transform=transform, download=download)
        if idx is None:
            return
        self.data = self.data[idx]
        targets_array = np.array(self.targets)
        self.targets = targets_array[idx].tolist()


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
n_train = 45_000
train_set = CIFAR10Set(idx=range(n_train), root="./data", train=True, transform=transform_train, download=True)
val_set = CIFAR10Set(idx =range(n_train, 50000), root="./data", train=True, transform=transform_eval, download=True)
test_set = CIFAR10Set(root="./data", train=True, transform=transform_eval, download=True)

#  print(train_set[20001][0].shape)
#  print(train_set[20001][1])
#  print(test_set[333][0].shape)
#  print(test_set[333][1])

dataloaders = {}
dataloaders["train"] = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
dataloaders["val"] = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
dataloaders["test"] = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#  print(device)

# for x, y in dataloaders["train"]:
#     print(x.shape, y)
#     break;



# ==================================== 1. Dropout

def my_dropout(X, p):
    assert 0 <= p <= 1
    X = X.float()
    if p == 1:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < 1 - p).float()
    return X * mask / (1 - p)

class Dropout(nn.Module):
    """
    Args:
        p: float, dropout probability
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        """
        Args:
            input: PyTorch tensor, arbitrary shape
        Returns:
            PyTorch tensor, same shape as input
        """
        if self.training:
            mask = input.new_empty(input.shape)
            mask.bernoulli_(1 - self.p)
            return mask * input / (1 - self.p)
        return input


#  X = torch.arange(9).reshape(3, 3)
#  X.bernoulli(0.5)
#  print(X)
#  X.bernoulli_(0.5)
#  print(X)
#  torch.bernoulli(X, 0.5)
#  for i in range(10):
#      X_drop = Dropout(0.5)(X)
#      print(X_drop)
#
#  # Test dropout
#  test = torch.rand(10_000)
#  dropout = Dropout(0.2)
#  test_dropped = dropout(test)
#
#  # These assertions can in principle fail due to bad luck, but
#  # if implemented correctly they should almost always succeed.
#  assert np.isclose(test_dropped.mean().item(), test.mean().item(), atol=1e-2)
#  assert np.isclose((test_dropped > 0).float().mean().item(), 0.8, atol=1e-2)

# ============================================ 2. Batch normalization
class BatchNorm(nn.Module):
    """
    Args: num_features, Number of features to calculate batch statistics for.
    """
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.parameter.Parameter(torch.ones(num_features))
        self.beta = nn.parameter.Parameter(torch.zeros(num_features))

    def forward(self, input):
        """
        Args:
            input: Pytorch tensor, shape (N, C, L)
        Return:
            Pytorch tensor, shame shape as input
        """
        eps = 1e-5
        mean = torch.mean(input, dim=[0, 2], keepdim=True)
        std = torch.std(input, dim=[0, 2], keepdim=True)

        input_normalized = (input - mean) / (std + eps)
        return input_normalized * self.gamma[None, :, None] + self.beta[None, :, None]

#  torch.random.manual_seed(42)
#  test = torch.randn(8, 2, 4)
#
#  b1 = BatchNorm(2)
#  test_b1 = b1(test)
#  b2 = nn.BatchNorm1d(2)
#  test_b2 = b2(test)
#
#  assert torch.allclose(test_b1, test_b2, rtol=0.2)

# ============================================ 3. ResNet
class Oper(nn.Module):
    def __init__(self, func=None):
        super().__init__()
        self.func = func

    def forward(self, x):
        if self.func is not None:
            return self.func(x)
        return x



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

#  for x, y in dataloaders["train"]:
#      x, y = x.to(device), y.to(device)
#      block = ResidualBlock(3, 16, 2)
#      block.to(device)
#      x_new = block(x)
#      print(x.shape)
#      print(x_new.shape)
#      break


class ResidualStack(nn.Module):
    """
    A stack of resudual blocks.

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        stride: Stride size of the first layer, used for downsampling
        num_blocks: Number of residual blocks
    """
    #  def __init__(self, in_channels, out_channels, stride, num_blocks):
    #      super().__init__()
    #      self.block1 = ResidualBlock(in_channels, out_channels, stride)
    #      self.block2 = ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1)
    #
    #  def forward(self, input):
    #      x = self.block2(self.block1(input))
    #      return x

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

#  for x, y in dataloaders["train"]:
#      x, y = x.to(device), y.to(device)
#      stack = ResidualStack(in_channels=3, out_channels=16, stride=2, num_blocks=5)
#      stack.to(device)
#      print(x.shape)
#      x_new = stack(x)
#      print(x.shape)
#      print(x_new.shape)
#      break

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

#  # target output size of 5x7
#  m = nn.AdaptiveMaxPool2d((5,7))
#  input = torch.randn(1, 64, 8, 9)
#  output = m(input)
#  print(output.shape)
#  # target output size of 7x7 (square)
#  m = nn.AdaptiveMaxPool2d(7)
#  input = torch.randn(1, 64, 10, 9)
#  output = m(input)
#  print(output.shape)
#  # target output size of 10x7
#  m = nn.AdaptiveMaxPool2d((None, 7))
#  input = torch.randn(1, 64, 10, 9)
#  output = m(input)
#  print(output.shape)

#  for x, y in dataloaders["train"]:
#      print(x.shape)
#      x_new = resnet(x)
def initialize_weight(module): 
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d): 
        nn.init.constant_(module.weight, 1) 
        nn.init.constant_(module.bias, 0)

resnet.apply(initialize_weight)

print(device)
resnet.to(device)
#  print(resnet)

#  for x, y in dataloaders["train"]:
#      x, y = x.to(device), y.to(device)
#      print(x.shape)
#      x_new = resnet(x)
#      print(x.shape)
#      print(x_new.shape)
#      break

# ================================================ 4. Training

def run_epoch(model, optimizer, dataloader, train):
    """ 
    Run one epoch of training or evaluation. 

    Args: 
        model: The model used for prediction 
        optimizer: optimization algorithm for the model 
        dataloader: Dataloader providing the data to run our model on 
        train: Whether this epoch is used for training or evaluation 

    Returns: 
        Loss and accuracy for this epoch
    """
    
    if train:
        model.train() 
    else: 
        model.eval() 

    loss_sum = 0.0 
    num_correct_sum = 0.0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device) 
        if train:
            optimizer.zero_grad() 

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        preds = torch.argmax(logits, dim=1)
        num_correct = (preds == yb).sum()

        if train: 
            loss.backward() 
            optimizer.step()

        loss_sum += loss.item() 
        num_correct_sum += num_correct
    #  print(loss_sum / len(dataloader.dataset), num_correct_sum / len(dataloader.dataset))
    return loss_sum / len(dataloader.dataset), num_correct_sum / len(dataloader.dataset)

resnet.to(device)
#  run_epoch(resnet, None, dataloaders["val"], train=False)


def fit(model, optimizer, lr_scheduler, dataloaders, max_epochs, patience): 
    """ 
    Fit the given model on the dataset 

    Args: 
        model: The model used for prediction 
        optimizer: optimization algorithm for the model 
        lr_scheduler: Learning rate scheduler that improves training in late epochs with learning rate decay 
        dataloaders: Dataloaders for training and validation 
        max_epochs: Maximum number of epochs for training 
        patience: Number of epochs to wait with early stopping the training if validation accuracy has decreased 
    """
    
    best_acc = 0
    curr_patience = 0
    best_weights = None
    for epoch in range(max_epochs): 
        train_loss, train_acc = run_epoch(model, optimizer, dataloaders["train"], train=True)
        print(f"Epoch {epoch + 1} / {max_epochs}, train loss: {train_loss:.2e}, accuracy: {train_acc * 100:.2f}%")

        val_loss, val_acc = run_epoch(model, optimizer, dataloaders["val"], train=False)
        print(f"Epoch {epoch + 1} / {max_epochs}, val loss: {val_loss:.2e}, accuracy: {val_acc * 100:.2f}%")

        lr_scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc 
            best_weights = copy.deepcopy(model.state_dict())
            curr_patience = 0
        else:
            curr_patience += 1

        if curr_patience >= patience:
            break 
        
        model.load_state_dict(best_weights)
        




optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 50], gamma=0.1)
fit(resnet, optimizer, lr_scheduler, dataloaders, max_epochs=200, patience=5)

