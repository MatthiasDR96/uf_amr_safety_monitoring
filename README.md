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
docker run --gpus all --runtime nvidia --rm -it --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix username/my-app:latest 
```
## Usage

Once the Docker container runs on the remote host, the inference starts and the result is visible via the webserver on host:8000. 

* Data-collection: Data can be collected using the record_video.py and playback_video.py scripts. The record_video.py script records a video using the ZED camera and saves the .svo file to the file you specify. After recording, the playback_video.py script playbacks the recorded video and image captures can be made that generates a database of images from the video. 

* Labelling: For labelling, the raw images can be loaded to LabelStudio and labelled according to your preferences. The labels can be exported in YOLO format and saved under the 'data/detection' folder. 

* Training: After labelling, the 'train_yolo.py' script can be used to train a model. 

* Inference: The main.py script loads the customized model and inferes on the stream coming from the ZED camera. 

## Debugging

For debugging, you can use the Visual Studio Code’s Remote - SSH extension to connect your local machine to your remote machine and use it as a development environment. This allows you to write code on your local machine, but run it on the remote machine. Your code from the local machine can be copied to the remote machine in a folder called e.g. uf_amr_safety_monitoring via ssh. In the command terminal, cd to your code repository and run the code below:

```bash
scp -r ./scripts/ nano@hostname:uf_amr_safety_monitoring/scripts
```

When starting the docker container, mount the working volume consisting of the files in /scripts on the remote machine to the folder in the docker container where these files are located, e.g. /uf_amr_safety_monitoring/scripts. 

```bash
sudo docker run --gpus all --runtime nvidia --rm -it --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/uf_amr_safety_monitoring  username/my-app:latest 
```

When running the container, the files will now be synced with the ones on the remote machine which can be modified from the local machine. 