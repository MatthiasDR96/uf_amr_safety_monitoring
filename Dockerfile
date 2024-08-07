# Using the base image with LT4-Pytorch for Jetpack 5.1.1 and ZED SDK
FROM stereolabs/zed:4.1-devel-jetson-jp5.1.1

# Install nano editor
#RUN apt-get install nano

# Set the working directory 
WORKDIR /uf_amr_safety_monitoring

# Copy the necessary files and directories into the container
COPY data/ /uf_amr_safety_monitoring/data/
COPY models/ /uf_amr_safety_monitoring/models/
COPY scripts/ /uf_amr_safety_monitoring/scripts/
COPY requirements.txt yolov8n.pt /uf_amr_safety_monitoring/

# Set file permissions
RUN chmod +x /uf_amr_safety_monitoring/scripts/main.py

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

