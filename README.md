# Ultimate Factory AMR Safety Monitoring

## Installation

Clone this repository to your local working directory.

```bash
git clone https://github.com/MatthiasDR96/uf_amr_safety_monitoring.git
```

Download and install Docker on your local machine. Build a Docker image and push it to the Docker registry.

```bash
docker login
docker build -t my-app:latest .  
docker tag my-app:latest username/my-app:latest 
docker push username/my-app:latest     
```

On the remote host, pull the image

```bash
ssh user@remote-server
docker login
docker pull username/my-app:latest
docker run --gpus all --runtime nvidia --device="/dev/video0:/dev/video0" username/my-app:latest 
```
## Usage

Once the Docker container runs on the remote host, the inference starts and the result is displayed. 

* Data-collection: Data can be collected using the record_video.py and playback_video.py scripts. The record_video.py script records a video using the ZED camera and saves the .svo file to the file you specify. After recording, the playback_video.py script playbacks the recorded video and image captures can be made that generates a database of images from the video. 

* Labelling: For labelling, the raw images can be loaded to LabelStudio and labelled according to your preferences. The labels can be exported in YOLO format and saved under the 'data/detection' folder. 

* Training: After labelling, the 'train_yolo.py' script can be used to train a model. 

* Inference: The main.py script loads the customized model and inferes on the stream coming from the ZED camera. 