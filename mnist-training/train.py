from aimstack.experiment_tracker import TrainingRun
from aimstack.base import Figure, FigureSequence, Metric, Image, ImageSequence
from aimstack.base.types.image import convert_to_aim_image_list

import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Initialize a new Run
aim_run = TrainingRun(repo='aim://0.0.0.0:8274')

# Device configuration
device = torch.device('cpu')

# Hyper parameters
num_epochs = 2
num_classes = 10
batch_size = 16
learning_rate = 0.03

# aim - Track hyper parameters
aim_run['hparams'] = {
    'num_epochs': num_epochs,
    'num_classes': num_classes,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
}

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_loss = Metric(aim_run, name='loss', context={'subset': 'train'})
# val_loss = Metric(aim_run, name='loss', context={'subset': 'val'})
train_acc = Metric(aim_run, name='acc', context={'subset': 'train'})
# val_acc = Metric(aim_run, name='acc', context={'subset': 'val'})
aim_images_s = ImageSequence(aim_run, name='images', context={'subset': 'train'})
fig_1 = FigureSequence(aim_run, name='figs1', context={'subset': 'train'})
fig_2 = FigureSequence(aim_run, name='figs2', context={'subset': 'train'})
fig_3 = FigureSequence(aim_run, name='figs3', context={'subset': 'train'})


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        aim_images = convert_to_aim_image_list(images, labels)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        import numpy
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], '
                  'Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                        total_step, loss.item()))
            train_loss.track(loss.item())

            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

            # aim - Track metrics
            train_acc.track(acc)

            if i % 200 == 0:
                aim_images_s.track(aim_images)
                # Track confusion matrix
                matrix_confusion = confusion_matrix(labels.detach().numpy(), predicted.detach().numpy(), range(10))
                z = matrix_confusion
                z_text = [[str(y) for y in x] for x in z]
                x = list(range(z.shape[0]))
                y = list(range(z.shape[1]))
                fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='reds')
                fig.update_layout(width=500, height=500)
                fig['data'][0]['showscale'] = True
                fig_3.track(Figure(fig))

                # Track table
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                fig_table = go.Figure(data=[go.Table(header=dict(values=['True', 'False'], height=34, font=dict(size=22)),
                                               cells=dict(values=[[correct], [total-correct]], height=34, font=dict(size=22)))
                                            ])
                fig_1.track(Figure(fig_table))

                # Track ROC AUC curves
                fig_line = go.Figure()
                for c in range(10):
                    pos_cls = []
                    for b in range(batch_size):
                        pos_cls.append(outputs.detach().numpy()[b][c])
                    fpr, tpr, thresholds = roc_curve(labels.detach().numpy(), pos_cls, pos_label=c)
                    df = pd.DataFrame({
                        'Thresholds': thresholds,
                        'FPR': fpr,
                        'TPR': tpr,
                    })

                    fig_line.add_trace(go.Line(x=df['FPR'], y=df['TPR'],
                                               name='ROC curve of class {}'.format(c)))
                fig_2.track(Figure(fig_line))

