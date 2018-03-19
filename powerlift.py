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
pd.options.display.float_format = '{:.4f}'.format

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

    processed_features["Age"] = linear_scale(processed_features["Age"])
    processed_features["BodyweightKg"] = linear_scale(processed_features["BodyweightKg"])
    processed_features["BestSquatKg"] = linear_scale(processed_features["BestSquatKg"])
    processed_features["BestBenchKg"] = linear_scale(processed_features["BestBenchKg"])

    return processed_features


def preprocess_targets(dataframe):
    output_targets = pd.DataFrame()
    output_targets["deadlift"] = dataframe["BestDeadliftKg"]

    output_targets["deadlift"] = linear_scale(output_targets["deadlift"])

    return output_targets

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

training_examples = preprocess_features(powerlifting_dataframe.head(12000))
training_targets = preprocess_targets(powerlifting_dataframe.head(12000))

validation_examples = preprocess_features(powerlifting_dataframe.tail(5000))
validation_targets = preprocess_targets(powerlifting_dataframe.tail(5000))

print(training_examples.head(100))
print(validation_targets.head(100))

