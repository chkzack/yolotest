# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Perform test request
"""

import pprint
import requests

DETECTION_URL = 'http://localhost:5000/v1/object-detection/yolov5s'
IMAGE_URL = 'https://learnsmart.edu.hk/wp-content/uploads/elementor/thumbs/pexels-photo-6039820-punakfxd9o0cpm3po6ook9oc166178gpmr95d8x1j6.jpg'
IMAGE = 'zidane.jpg'

# Image file upload
# Read image
with open(IMAGE, 'rb') as f:
    image_data = f.read()

response = requests.post(DETECTION_URL, files={'image': image_data}).json()
pprint.pprint(response)

# Remote image url
response = requests.post(DETECTION_URL, json={'image_url': IMAGE_URL}).json()
pprint.pprint(response)
