import os

# Define the directory structure
directories = [
    "weight",
    "result",
    "data",
    "scripts"
]

# Define the files to create with empty content
files = {
    "data/dataloaders.py": "",
    "data/_init__.py": "",
    "train.py": "",
    "config.py": "",
    "run.py": "",
    "_init__.py": "",
    "scripts/inference.py": "",
    "scripts/eval.py": "",
    "scripts/test.py": ""
}
try:
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)  # Create directories if they don't already exist

    # Create files with empty content
    for file_path in files:
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create the file with empty content
        with open(file_path, 'w') as file:
            pass  # Write nothing, just create the file

        print("Directories and empty files created successfully.")

except Exception as e:
    print(f"An error occurred: {e}")

# print("Directories and empty files created successfully.")
