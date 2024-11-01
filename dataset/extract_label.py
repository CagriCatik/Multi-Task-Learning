import os

# Define mappings for gender and race
gender_map = {0: 'male', 1: 'female'}
race_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}

# Example filename: (age_gender_race_timestamp.jpg)
filename = "./UTKFace/50_0_1_20210101010101.jpg"

# Extract just the filename (without directory path)
basename = os.path.basename(filename)

# Split the basename to extract age, gender, and race
age, gender, race, _ = basename.split("_")

# Convert age, gender, and race to integers
age = int(age)
gender = int(gender)
race = int(race)

# Map gender and race to their respective labels
gender_label = gender_map[gender]
race_label = race_map[race]

# Output the extracted and mapped data
print(f"Age: {age}")
print(f"Gender: {gender_label}")
print(f"Race: {race_label}")
