import pandas as pd
import os

input_path = "./Dataset/NGSIM/trajectories-0750am-0805am.txt"
output_path = "./Dataset/NGSIM/US101_cleaned.csv"

cols = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "Vehicle_Length", "Vehicle_Width", "Vehicle_Class",
    "Local_Velocity", "Local_Accel", "Lane_ID",
    "Preceding", "Following", "Space_Headway",
    "Time_Headway", "Location", "Section_ID",
    "Direction", "Movement", "Origin", "Destination"
]

#  Correctly parse using whitespace delimiter
df = pd.read_csv(input_path, delim_whitespace=True, names=cols)

# Keep only the useful subset
df_clean = df[["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "Local_Velocity", "Lane_ID"]]
df_clean = df_clean.dropna().astype(float)
df_clean = df_clean[df_clean["Lane_ID"] > 0]  # remove invalid 0s

os.makedirs("./Dataset/NGSIM", exist_ok=True)
df_clean.to_csv(output_path, index=False)
print(" Cleaned CSV saved to:", output_path)
print(df_clean.head())
