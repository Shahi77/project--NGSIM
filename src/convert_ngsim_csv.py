# convert_ngsim_csv.py (fixed)
import pandas as pd, os

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

df = pd.read_csv(input_path, delim_whitespace=True, names=cols)

#  keep all 7 core features
df_clean = df[["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y",
               "Local_Velocity", "Local_Accel", "Lane_ID"]]
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean["Lane_ID"] > 0]
df_clean = df_clean.astype(float)

os.makedirs("./Dataset/NGSIM", exist_ok=True)
df_clean.to_csv(output_path, index=False)
print(" Cleaned CSV saved to:", output_path)
print(df_clean.head())
