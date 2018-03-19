import pandas as pd

powerlifting_file = open("powerlifting-database/openpowerlifting.csv", "rb")
powerlifting_dataframe = pd.read_csv(powerlifting_file, sep=",")

powerlifting_dataframe = powerlifting_dataframe.drop(["MeetID", "Name", "Division", "WeightClassKg", "Squat4Kg", \
                                                      "Bench4Kg", "Deadlift4Kg", "TotalKg", "Place", "Wilks"], 1)
powerlifting_dataframe = powerlifting_dataframe.dropna()
sex_mapping = {'M': 0.0, 'F': 1.0}
equipment_mapping = {'Wraps': 0.0, 'Single-ply': 1.0, 'Raw': 2.0, 'Multi-ply': 3.0}

powerlifting_dataframe = powerlifting_dataframe.replace({'Sex': sex_mapping, 'Equipment': equipment_mapping})
powerlifting_dataframe.to_csv("cleanedlifting.csv", sep=',', encoding='utf-8')



