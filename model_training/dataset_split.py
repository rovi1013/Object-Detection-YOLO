import os
import shutil
from sklearn.model_selection import train_test_split

# Parameters
dataset_dir = './annotations'
output_dir = './dataset'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
random_seed = 42

# Create output directory structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)


# Function to copy files to their respective directories
def copy_files(dataset_split, split_name):
    for image_file, label_file in dataset_split:
        shutil.copy(image_file, os.path.join(output_dir, 'images', split_name, os.path.basename(image_file)))
        shutil.copy(label_file, os.path.join(output_dir, 'labels', split_name, os.path.basename(label_file)))


# Process each subfolder in the dataset_dir
for subfolder in os.listdir(dataset_dir):
    subfolder_path = os.path.join(dataset_dir, subfolder)
    if os.path.isdir(subfolder_path):
        image_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if
                       f.endswith(('.png', '.jpg'))]
        label_files = [f.replace('.png', '.txt').replace('.jpg', '.txt') for f in image_files]

        # Combine images and labels into a single list
        data = [(img, lbl) for img, lbl in zip(image_files, label_files) if os.path.exists(lbl)]

        if len(data) == 0:
            print(f"Warning: No valid data found in subfolder {subfolder}. Skipping this subfolder.")
            continue

        try:
            # Split data into train, val, and test
            train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio), random_state=random_seed)
            val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)),
                                                   random_state=random_seed)

            # Copy files to train, val, and test directories
            copy_files(train_data, 'train')
            copy_files(val_data, 'val')
            copy_files(test_data, 'test')
        except ValueError as e:
            print(f"Error in subfolder {subfolder}: {e}")

print("Dataset has been split and copied successfully.")
