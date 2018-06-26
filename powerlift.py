import os
import math
import numpy as np
import pandas as pd
from sklearn import metrics

import tensorflow as tf
from tensorflow.python.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.2f}'.format

powerlifting_file = open("powerlifting-database/cleanedlifting.csv", "rb")
powerlifting_dataframe = pd.read_csv(powerlifting_file, sep=",")

# Remove rows with negative values
powerlifting_dataframe = powerlifting_dataframe[(
    powerlifting_dataframe >= 0).all(1)]

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


training_examples = preprocess_features(powerlifting_dataframe.head(12000))
training_targets = preprocess_targets(powerlifting_dataframe.head(12000))

validation_examples = preprocess_features(powerlifting_dataframe.tail(5000))
validation_targets = preprocess_targets(powerlifting_dataframe.tail(5000))


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(
        my_optimizer,
        steps,
        hidden_units,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    nn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        activation_fn=tf.nn.relu,
    )

    # Create input functions
    def training_input_fn(): return my_input_fn(training_examples,
                                                training_targets["deadlift"],
                                                batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(training_examples,
                                                        training_targets["deadlift"],
                                                        num_epochs=1,
                                                        shuffle=False)

    def predict_validation_input_fn(): return my_input_fn(validation_examples,
                                                          validation_targets["deadlift"],
                                                          num_epochs=1,
                                                          shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        # Train the model, starting from the prior state.
        nn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = nn_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])

        validation_predictions = nn_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # Occasionally print the current loss.
        print("  period %02d : %0.8f" %
              (period, training_root_mean_squared_error))

        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished.")

    print("Final RMSE (on training data):   %0.8f" %
          training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.8f" %
          validation_root_mean_squared_error)

    return nn_regressor, training_rmse, validation_rmse


model, _, _ = train_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    steps=200,
    hidden_units=[5, 5],
    batch_size=32,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


def result_for_one(model, ex_orig, ex_feat, ex_targ):
    ex_orig.drop(ex_orig.columns[0], 1, inplace=True)

    def ex_input_fn(): return my_input_fn(ex_feat,
                                          ex_targ["deadlift"],
                                          num_epochs=1,
                                          shuffle=False)

    validation_predictions = model.predict(input_fn=ex_input_fn)
    validation_predictions = np.array(
        [item['predictions'][0] for item in validation_predictions])

    error = metrics.mean_absolute_error(validation_predictions, ex_targ)

    print("\nThe target was: {}\n".format(str(ex_targ["deadlift"].item())))
    print("The predicted value was: {}".format(str(validation_predictions)))
    print("The error for this example is: {}".format(str(error)))
    print("The features were: {}\n".format(ex_orig.to_string(index=False)))


powerlifting_dataframe.hist()

ex_orig = powerlifting_dataframe.tail(1).copy()
ex_feat = validation_examples.tail(1).copy()
ex_targ = validation_targets.tail(1).copy()

result_for_one(model, ex_orig, ex_feat, ex_targ)
