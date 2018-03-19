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

powerlifting_file = open("powerlifting-database/openpowerlifting.csv", "rb")
powerlifting_dataframe = pd.read_csv(powerlifting_file, sep=",")

powerlifting_dataframe = powerlifting_dataframe.reindex(
    np.random.permutation(powerlifting_dataframe.index))

powerlifting_dataframe = powerlifting_dataframe.dropna()


def preprocess(powerlifting_dataframe):
    selected_features = powerlifting_dataframe[
        ["Sex",
         "Equipment",
         "Age",
         "Bodyweight",
         "BestSquatKg",
         "BestBenchKg",
         "BestDeadliftKg"]]

