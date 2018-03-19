import math
import os

# from matplotlib import cm
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
# import tensorflow as tf
# from tensorflow.python.data import Dataset

# tf.logging.set_verbosity(tf.logging.ERROR
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

powerlifting_file = open("powerlifting-database/cleanedlifting.csv", "rb")
powerlifting_dataframe = pd.read_csv(powerlifting_file, sep=",")

powerlifting_dataframe = powerlifting_dataframe.reindex(
    np.random.permutation(powerlifting_dataframe.index))


def preprocess_features(dataframe):
    selected_features = dataframe[
        ["Sex",
         "Equipment",
         "Age",
         "BodyweightKg",
         "BestSquatKg",
         "BestBenchKg"]]
    processed_features = selected_features.copy()
    return processed_features


def preprocess_targets(dataframe):
    output_targets = pd.DataFrame()
    output_targets["deadlift"] = dataframe["BestDeadliftKg"]
    return output_targets


training_examples = preprocess_features(powerlifting_dataframe.head(12000))
training_targets = preprocess_targets(powerlifting_dataframe.head(12000))

validation_examples = preprocess_features(powerlifting_dataframe.tail(5000))
validation_targets = preprocess_targets(powerlifting_dataframe.tail(5000))

print(validation_targets.head(3))




