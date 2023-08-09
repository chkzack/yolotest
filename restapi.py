# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
import cv2
import torch
from flask import Flask, request
from PIL import Image
from urllib.parse import urlparse

app = Flask(__name__)
# models = {}
model = NotImplemented

DETECTION_URL = '/v1/object-detection/<model>'


@app.route(DETECTION_URL, methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    im = NotImplemented
    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        # if model in models:
        #     results = models[model](im, size=640)  # reduce size=320 for faster inference
        #     return results.pandas().xyxy[0].to_json(orient='records')
    else:
        # Method 3
        img_url = request.json.get('image_url')
        p = urlparse(img_url)
        full_file_name = p.path.rsplit("/", 1)[-1]
        file_name = full_file_name.split('.')[0]
        torch.hub.download_url_to_file(img_url, file_name)  # download 2 images
        im = Image.open(file_name)  # PIL image
        # im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    
    if im != NotImplemented and model != NotImplemented:
        results = model(im, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient='records')
    else:
        return "error: image or model load error"



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    # for m in opt.model:
    #     models[m] = torch.hub.load('ultralytics/yolov5', m, force_reload=True, skip_validation=True)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='/usr/yolov5s.pt', force_reload=True, skip_validation=True)
    model = torch.load('/usr/yolov5s.pt', map_location=lambda storage, loc: storage.cuda(0))

    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat
