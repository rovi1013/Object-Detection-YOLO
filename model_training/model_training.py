import ultralytics
import torch
import multiprocessing as mp


def main():
    # Step 0: Check for CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load the YOLO model
    model = ultralytics.YOLO(f'yolov9t.pt')  # YOLO model version and variant can be *.pt or *.yaml

    # Step 2: Train the model with additional configurations
    model.train(
        data='./birds.yml',     # Path to dataset configuration file
        epochs=150,             # Number of epochs
        batch=16,               # Batch size for training
        imgsz=640,              # Image size
        name='bird_model',      # Name of the custom model
        patience=50,            # Early stopping patience
        lr0=0.001,              # Initial learning rate
        lrf=0.1,                # Final OneCycleLR learning rate (fraction of lr0)
        momentum=0.937,         # SGD momentum
        weight_decay=0.0005,    # optimizer weight decay
        warmup_epochs=3.0,      # warmup epochs (fractions ok)
        warmup_momentum=0.8,    # warmup initial momentum
        warmup_bias_lr=0.1,     # warmup initial bias lr
        cos_lr=True,            # Use cosine annealing learning rate
        box=0.05,               # box loss gain
        cls=0.5,                # cls loss gain
        hsv_h=0.015,            # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,              # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,              # image HSV-Value augmentation (fraction)
        degrees=0.0,            # image rotation (+/- deg)
        translate=0.1,          # image translation (+/- fraction)
        scale=0.5,              # image scale (+/- gain)
        shear=0.0,              # image shear (+/- deg)
        perspective=0.0,        # image perspective (+/- fraction), range 0-0.001
        flipud=0.0,             # image flip up-down (probability)
        fliplr=0.5,             # image flip left-right (probability)
        mosaic=1.0,             # image mosaic (probability)
        mixup=0.2,              # image mixup (probability) for better generalization
        copy_paste=0.1,         # segment copy-paste (probability) for better augmentation
        close_mosaic=10         # Close mosaic augmentation after this epoch for stability
    )


if __name__ == '__main__':
    mp.freeze_support()
    main()
