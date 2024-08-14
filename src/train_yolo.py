# Imports
import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Set directory paths
data_dir = str(Path('./data/dataset.yaml').resolve())
save_dir = './models'
project_dir = './models'

# Set epochs
epochs = 100

# Remove previous model
shutil.rmtree('./models/train', ignore_errors=True)

# Finetune model
model.train(epochs=epochs, data=data_dir, project=project_dir, cache=False, val=False, save_dir=save_dir)

# Load a finetuned YOLOv8n model
model = YOLO('./models/train/weights/best.pt')

# Predict with the model
for image_path in os.listdir('./data/images')[0:5]:

    # Read frame
    frame = cv2.imread('./data/images/' + image_path)

    # Predict
    results = model(frame, stream=True, verbose=False, conf=0.01, iou=0.01)

    # Plot results
    for result in results:
        im = result.show()  # plot a BGR numpy array of predictions
