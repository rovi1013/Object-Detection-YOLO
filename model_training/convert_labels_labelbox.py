import json
import os
import shutil

# Define the input file path and output directories
input_file_path = 'labelbox_annotation.ndjson'
images_base_dir = '../download_images/birds_dataset'
output_dir = 'annotations'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


def parse_labelbox_ndjson(ndjson_path):
    with open(ndjson_path, 'r') as f:
        return [json.loads(line) for line in f]


def convert_annotations_to_yolo(labelbox_annotations, output_directory, images_directory):
    for item in labelbox_annotations:
        image_name = item['data_row']['external_id']
        image_path = os.path.join(images_directory, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist.")
            continue

        img_width = item['media_attributes']['width']
        img_height = item['media_attributes']['height']
        project_data = item['projects']
        yolo_annotations = []

        for project_id, project_info in project_data.items():
            for label in project_info['labels']:
                for obj in label['annotations']['objects']:
                    if 'bounding_box' in obj:
                        bbox = obj['bounding_box']
                        class_id = 0  # Assuming all are 'blue_tit', adjust class_id mapping as needed
                        x_center = (bbox['left'] + bbox['width'] / 2) / img_width
                        y_center = (bbox['top'] + bbox['height'] / 2) / img_height
                        width = bbox['width'] / img_width
                        height = bbox['height'] / img_height

                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Save YOLO annotation file
        annotation_file = os.path.join(output_directory, os.path.splitext(image_name)[0] + '.txt')
        with open(annotation_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        # Copy image to output directory
        shutil.copy(image_path, os.path.join(output_directory, image_name))


# Step 1: Read and parse the NDJSON file
annotations = parse_labelbox_ndjson(input_file_path)

# Step 2: Process each subfolder in the images_base_dir
for subfolder in os.listdir(images_base_dir):
    subfolder_path = os.path.join(images_base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        images_dir = subfolder_path
        subfolder_output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(subfolder_output_dir, exist_ok=True)

        # Step 3: Convert and save YOLOv8 annotations for the current subfolder
        convert_annotations_to_yolo(annotations, subfolder_output_dir, images_dir)

print("Conversion to YOLO format completed for all subfolders.")
