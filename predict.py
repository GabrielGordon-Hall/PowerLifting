import random
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataloading import CSVDataset, ToTensor


def predict(model, input):
    features, target = input['features'], input['targets']
    print("Stats are {}".format(features))
    features, target = Variable(
        features, volatile=True), Variable(target, volatile=True)
    model.eval()
    pred = model(features)
    print("Model predicts best deadlift is {}, actual value is {}".format(
        pred.data[0], target.data[0]))


dataframe = pd.read_csv('powerlifting-database/processedlifts.csv')
datapoint = dataframe.sample(1)
datapoint = CSVDataset(datapoint, transform=ToTensor())[0]


model = torch.load('trained-model.pt')
predict(model, datapoint)
