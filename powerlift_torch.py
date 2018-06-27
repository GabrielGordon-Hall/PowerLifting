import time
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataloading import CSVDataset, ToTensor


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

hyper = {
    'input_size': 10,
    'hidden_size_a': 100,
    'hidden_size_b': 50,
    'output_size': 1
}


datasets = {x: CSVDataset(datasets[x], transform=ToTensor())
            for x in ['train', 'val']}
dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}


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
            for batch in dataloaders[phase]:
                inputs, targets = batch['features'], batch['targets']
                # Zero the parameter gradients
                optimiser.zero_grad()

                # Only track history if training
                if phase == 'train':
                    inputs = Variable(inputs, requires_grad=True)
                else:
                    inputs = Variable(inputs, volatile=True)

                inputs = inputs.double()
                targets = Variable(targets).double()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimiser.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)

            epoch_loss = (running_loss / dataset_sizes[phase])
            print('{} Loss: {:.6f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = c(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:6f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


model = nn.Sequential(
    nn.Linear(hyper[
        'input_size'], hyper['hidden_size_a']),
    nn.ReLU(),
    nn.Linear(hyper['hidden_size_a'], hyper['output_size']),
).double()

criterion = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

model_ft = train_model(model, criterion, optimiser, exp_lr_scheduler,
                       num_epochs=10)
torch.save(model_ft, 'trained-model.pt')
