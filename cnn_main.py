import torch
import torch.nn as nn
from tqdm.auto import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets

from torchmetrics import Accuracy

#linear_layer = nn.Linear(5, 3, bias = True)
#activation = nn.Tanh()

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.accuracy = Accuracy(task="multiclass", num_classes=10)

    self.l1 = nn.Conv2d(1, 10, kernel_size=5, stride = 1)
    self.l2 = nn.Conv2d(10, 10, kernel_size=5, stride = 1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.l4 = nn.Linear(4*4*10, 100)
    self.l5 = nn.Linear(100, 10)

  def forward(self, x):

    x = self.l1(x)
    x = nn.functional.relu(x)
    x = self.pool(x)
    x = self.l2(x)
    x = nn.functional.relu(x)
    x = self.pool(x)
    x = x.view(-1, 4*4*10)
    x = self.l4(x)
    x = nn.functional.relu(x)
    x = self.l5(x)

    return x

  def my_train(self, train_dataloader, criterion, optimizer, num_epoch, dev):
    self.train()
    for t in tqdm(range(num_epoch), desc = "Epochs"):
      temp_loss = temp_acc = 0.0
      for x, y in train_dataloader:
        x, y = x.to(dev), y.to(dev)
        y_pred = self(x)
        loss = criterion(y_pred, y)
        acc = self.accuracy(y_pred, y)

        temp_loss += loss.item() * x.size(0)
        temp_acc += acc.item() * x.size(0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

      self.accuracy.reset()
      epoch_loss = temp_loss/len(train_dataloader.dataset)
      epoch_acc = temp_acc/len(train_dataloader.dataset)
      print("Epoch:", t, "Loss:", epoch_loss, "Accurancy:", epoch_acc)

    return self


  def test(self, test_dataloader, dev):
    self.eval()
    acc = 0.0
    with torch.no_grad():
      for x, y in test_dataloader:
        x, y = x.to(dev), y.to(dev)
        y_pred = self(x)
        acc += self.accuracy(y_pred, y) * x.size(0)
    print("Accurency on tests:", (acc/len(test_dataloader.dataset)).item())
    self.accuracy.reset()
    return

training_data = datasets.MNIST(
    root="data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = False)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(dev)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.my_train(train_dataloader, criterion, optimizer, 10, dev)
model.test(test_dataloader, dev)

