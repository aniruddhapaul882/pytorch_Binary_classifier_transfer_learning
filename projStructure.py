import os

# Define the directory structure
directories = [
    "config",
    "data/raw",
    "data/processed",
    "data/splits",
    "models",
    "outputs/logs",
    "outputs/results",
    "outputs/predictions",
    "outputs/figures",
    "checkpoints",
    "scripts",
    "tests"
]

# Define the files to create with empty content
files = {
    "config/config.yaml": "",
    "config/default_config.py": "",
    "config/_init__.py": "",
    "data/data_preparation.py": "",
    "data/datasets.py": "",
    "data/transforms.py": "",
    "data/dataloaders.py": "",
    "data/_init__.py": "",
    "models/model.py": "",
    "models/train.py": "",
    "models/evaluate.py": "",
    "models/loss_functions.py": "",
    "models/optimizers.py": "",
    "scripts/run_training.py": "",
    "scripts/run_inference.py": "",
    "tests/test_model.py": "",
    "tests/test_data_preparation.py": ""
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
