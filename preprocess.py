import numpy as np
import pandas as pd

powerlifting_file = open("powerlifting-database/cleanedlifting.csv", "rb")
powerlifting_dataframe = pd.read_csv(powerlifting_file, sep=",")

# Remove rows with negative values
powerlifting_dataframe = powerlifting_dataframe[(
    powerlifting_dataframe >= 0).all(1)]


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
    processed_features["BodyweightKg"] = linear_scale(
        processed_features["BodyweightKg"])
    processed_features["BestSquatKg"] = linear_scale(
        processed_features["BestSquatKg"])
    processed_features["BestBenchKg"] = linear_scale(
        processed_features["BestBenchKg"])

    one_hot_sex = pd.get_dummies(processed_features["Sex"], prefix='sex')
    one_hot_equipment = pd.get_dummies(
        processed_features["Equipment"], prefix='equipment')

    processed_features = processed_features.drop(["Sex", "Equipment"], 1)
    processed_features = pd.concat(
        [processed_features, one_hot_equipment, one_hot_sex], axis=1)

    return processed_features


def preprocess_targets(dataframe):
    output_targets = pd.DataFrame()
    output_targets["deadlift"] = dataframe["BestDeadliftKg"]
    # output_targets["deadlift"] = linear_scale(output_targets["deadlift"])
    return output_targets


def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


features = preprocess_features(powerlifting_dataframe)
targets = preprocess_targets(powerlifting_dataframe)
dataset = pd.concat([features, targets], axis=1)
pd.DataFrame.sort_index(dataset, inplace=True)

dataset.to_csv('powerlifting-database/processedlifts.csv', index=False)
