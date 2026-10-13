import os
import shutil

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = r"C:\Users\sriin\Desktop\Dynamic_Ongoing\features"

TEST_DIR = os.path.join(BASE_DIR, "TEST CSV")
TRAIN_DIR = os.path.join(BASE_DIR, "Train_CSV")

# --------------------------------------------------
# MOVE 5 CSVs FROM EACH CLASS
# --------------------------------------------------

NUM_TO_MOVE = 5

classes = [
    "GOOD_AFTERNOON",
    "GOOD_EVENING",
    "GOOD_MORNING",
    "HELLO",
    "HI",
    "PLEASE",
    "SORRY",
    "THANK_YOU",
    "WELCOME",
    "WHAT"
]

for class_name in classes:

    source_folder = os.path.join(TEST_DIR, class_name)
    destination_folder = os.path.join(TRAIN_DIR, class_name)

    # Make sure destination class folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Get CSV files only
    csv_files = sorted([
        f for f in os.listdir(source_folder)
        if f.lower().endswith(".csv")
    ])

    # Select first 5
    files_to_move = csv_files[:NUM_TO_MOVE]

    print(f"\n{class_name}")
    print(f"Found: {len(csv_files)} CSVs")
    print(f"Moving: {len(files_to_move)} CSVs")

    for filename in files_to_move:

        source = os.path.join(source_folder, filename)
        destination = os.path.join(destination_folder, filename)

        shutil.move(source, destination)

        print(f"  Moved: {filename}")

print("\n====================================")
print("DONE")
print("50 CSV files should have been moved.")
print("====================================")