# The-usage-of-tensorboard
#### 96. torch.utils.tensorboard

TensorBoard is a visualization toolkit for machine learning experimentation. TensorBoard allows tracking and visualizing metrics such as loss and accuracy, visualizing the model graph, viewing histograms, displaying images and much more. In this tutorial we are going to cover TensorBoard installation, basic usage with PyTorch, and how to visualize data you logged in TensorBoard UI.



Let’s now try using TensorBoard with PyTorch! Before logging anything, we need to create a `SummaryWriter` instance.



**Writer will output to `./runs/` directory by default.**



##### 1. Log scalars

In machine learning, it’s important to understand key metrics such as loss and how they change during training. Scalar helps to save the loss value of each training step, or the accuracy after each epoch.

To log a scalar value, use `add_scalar(tag, scalar_value, global_step=None, walltime=None)`. For example, lets create a simple linear regression training, and log loss value using `add_scalar`

```python
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
writer.flush()
```

Call `flush()` method to make sure that all pending events have been written to disk.

See [torch.utils.tensorboard tutorials](https://pytorch.org/docs/stable/tensorboard.html) to find more TensorBoard visualization types you can log.

If you do not need the summary writer anymore, call `close()` method.

```
writer.close()
```

