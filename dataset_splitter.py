import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(src_dir, dest_dir, val_size=0.2):
    # Create the destination directories if they don't exist
    os.makedirs(os.path.join(dest_dir, 'train', 'harness'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'train', 'no_harness'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'val', 'harness'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'val', 'no_harness'), exist_ok=True)

    
    categories = ['harness', 'no_harness']

    counts = {
        'train_harness': 0,
        'train_no_harness': 0,
        'val_harness': 0,
        'val_no_harness': 0
    }

    for category in categories:
        category_path = os.path.join(src_dir, category)
        print(f"Processing category: {category}")
        print(f"Category path: {category_path}")

        if not os.path.exists(category_path):
            print(f"Directory not found: {category_path}")
            continue

        
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        
        
        train_files, val_files = train_test_split(files, test_size=val_size, random_state=42)

        
        for file in train_files:
            src_file = os.path.join(category_path, file)
            dest_file = os.path.join(dest_dir, 'train', category, file)
            shutil.copy(src_file, dest_file)
            counts[f'train_{category}'] += 1

        for file in val_files:
            src_file = os.path.join(category_path, file)
            dest_file = os.path.join(dest_dir, 'val', category, file)
            shutil.copy(src_file, dest_file)
            counts[f'val_{category}'] += 1

    print(f"Number of files in train/harness: {counts['train_harness']}")
    print(f"Number of files in train/no_harness: {counts['train_no_harness']}")
    print(f"Number of files in val/harness: {counts['val_harness']}")
    print(f"Number of files in val/no_harness: {counts['val_no_harness']}")


source_directory = 'Harness_dataset'
destination_directory = 'Harness_dataset_split'

if __name__ == "__main__":
    split_data(source_directory, destination_directory, val_size=0.2)
