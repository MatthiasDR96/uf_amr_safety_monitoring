# Using the base image with LT4-Pytorch for Jetpack 5.1.1 and ZED SDK
FROM stereolabs/zed:4.1-devel-jetson-jp5.1.1

# Set the working directory 
WORKDIR /uf_amr_safety_monitoring

# Copy the necessary files and directories into the container
COPY data/ /uf_amr_safety_monitoring/data/
COPY models/ /uf_amr_safety_monitoring/models/
COPY scripts/ /uf_amr_safety_monitoring/scripts/
COPY requirements.txt start.sh yolov8n.pt /uf_amr_safety_monitoring/

# Set file permissions
RUN chmod +x /uf_amr_safety_monitoring/scripts/main.py
RUN chmod +x /uf_amr_safety_monitoring/start.sh

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# To run the application directly
CMD ["./start.sh"]

