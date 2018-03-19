import numpy as np
import pandas as pd

powerlifting_file = open("powerlifting-database/openpowerlifting.csv", "rb")
powerlifting_dataframe = pd.read_csv(powerlifting_file, sep=",")

powerlifting_dataframe = powerlifting_dataframe.drop(["MeetID", "Name", "Division", "WeightClassKg", "Squat4Kg", \
                                                      "Bench4Kg", "Deadlift4Kg", "TotalKg", "Place", "Wilks"], 1)
powerlifting_dataframe = powerlifting_dataframe.dropna()
mapping = {'M': 0, 'F': 1}
powerlifting_dataframe["Sex"]

print(powerlifting_dataframe.info())



