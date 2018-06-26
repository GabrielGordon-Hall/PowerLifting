import math
import copy
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler


dataframe = pd.read_csv('powerlifting-database/processedlifts.csv')
dataframe = dataframe.sample(
    frac=1).reset_index(drop=True)

datasets = {
    'train': dataframe.head(12000),
    'val': dataframe.tail(5024)
}

params = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4
}

hyperparams = {
    'input_size': 10,
    'hidden_size': 5,
    'output_size': 1
}


class CSVDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx, :-1].values
        features = features.astype('float').reshape(1, 10)
        labels = self.dataframe.iloc[idx, 0]
        sample = {'features': features, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays to Tensors"""

    def __call__(self, sample):
        features, labels = sample['features'], sample['labels']
        return {'features': torch.from_numpy(features),
                'labels': torch.from_numpy(np.array(labels, ndmin=1, copy=False))}


datasets = {x: CSVDataset(datasets[x], transform=ToTensor())
            for x in ['train', 'val']}
dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return self.fc3(out)


def train_model(model, criterion, optimiser, scheduler, num_epochs=25):
    c = copy.deepcopy
    since = time.time()
    best_model_weights = c(model.state_dict())
    best_loss = (2 ** 63) - 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                # Zero the parameter gradients
                optimiser.zero_grad()

                # Forward
                # Only track history if training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = c(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


model = NeuralNet(hyperparams['input_size'],
                  hyperparams['hidden_size'], hyperparams['output_size'])
criterion = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

model_ft = train_model(model, criterion, optimiser, exp_lr_scheduler,
                       num_epochs=25)
