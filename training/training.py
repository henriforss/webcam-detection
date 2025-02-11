# Description: This script downloads the dataset from Roboflow, moves the data to appropriate folders, and trains the YOLOv5 model.
# Run on GPU

from roboflow import Roboflow
import shutil
from ultralytics import YOLO
import os
from dotenv import load_dotenv

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Read api key from .env file
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Load dataset from Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("tuomari").project("face-detection-ynmgf")
version = project.version(1)
dataset = version.download("yolov8")
folder_name = dataset.location.split("/")[-1]

# Move data to appropriate folders
shutil.move(f'{dataset.location}/test',
            f'{dataset.location}/{folder_name}/test')
shutil.move(f'{dataset.location}/train',
            f'{dataset.location}/{folder_name}/train')
shutil.move(f'{dataset.location}/valid',
            f'{dataset.location}/{folder_name}/valid')

# Train model
model = YOLO('yolov8n')
results = model.train(
    data=f'{dataset.location}/data.yaml', epochs=100, imgsz=640)
