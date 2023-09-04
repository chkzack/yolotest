# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import os
import argparse
import io
import cv2
import torch
from flask import Flask, request, send_file
from PIL import Image
from urllib.parse import urlparse
import time
import numpy
import datetime
import json
from json import JSONEncoder
# need python version >= 3.7
from enum import Enum


DETECTION_URL = '/v1/object-detection/scan'
TRMP_DIR = '/tmp/images/'

app = Flask(__name__)
models = {}

# 标签类型中英文对照
type_dict = {
    'grid_meter': '电网表计',
    'grid_insulator': '电网绝缘子',
}

class Code(Enum):
    '''
    错误码：

    正确:2000
    图像数据错误:2001
    算法分析失败:2002
    '''
    SUCCESS = '2000'
    DATA_ERR = '2001'
    PROGRAM_ERR = '2002'

class LogType(Enum):
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4

class Response:
    def __init__(self):
        self.objectId = ''     # 名称
        self.typeList = ["bj_bjmh", "bj_bjds"] # 识别类型
        self.imageUrlList = ["jpg图像"] # 图片解析格式
        self.results = []     # 结果列表

class Result:
    def __init__(self, type='', value='', code=Code.SUCCESS, position=[], conf=0.0, desc='', result_image_url=''):
        self.type = type  # 分析类型
        self.value = value # 值
        self.code = code.value
        self.pos = position
        self.conf = conf
        self.desc = desc
        self.resultImageUrl = result_image_url

class Position:
    def __init__(self):
        self.areas = []

class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

class ResponseEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def log(str, log_type=LogType.INFO):
    '''
    日志输出
    '''
    if log_type == LogType.DEBUG:
        app.logger.debug(str)
    elif log_type == LogType.INFO:
        app.logger.info(str)
    elif log_type == LogType.WARN:
        app.logger.warning(str)
    elif log_type == LogType.ERROR:
        app.logger.error(str)   


def response(results=[], code=Code.SUCCESS):
    '''
    响应封装
    '''
    response = Response()
    response.objectId = 'scan'
    if len(results) == 0:
        result = Result(code=code)
        response.results.append(result)
    else:
        response.results = results
    return json.dumps(response, cls=ResponseEncoder)


def predict_process(im, return_img=False):
    '''
    使用模型检测并封装返回对象
    '''
    try:
        results = models['scan'](im, size=640)  # reduce size=320 for faster inference
        predict_list = results.pandas().xyxy[0].values.tolist()
        
        log(results.pandas().xyxy[0], LogType.INFO)
        result_list = []
        for data in predict_list:
            position = Position()

            rect_point_start = Point()
            rect_point_start.x = int(data[0]) # x1
            rect_point_start.y = int(data[1]) # y1
            position.areas.append(rect_point_start)

            rect_point_end = Point()
            rect_point_end.x = int(data[2]) # x2
            rect_point_end.y = int(data[3]) # y2
            position.areas.append(rect_point_end)

            confidencel = data[4] # confidencel
            type_class = data[5] # class
            name = data[6] # name
            desc = type_dict.get(name)

            result = Result(type=name, conf=confidencel, desc=desc if desc != None else '', position=position)
            result_list.append(result)

        log('scan image ok' + str(datetime.datetime.now()), LogType.INFO)
        return response(result_list)
    except Exception as error:
        log(error, LogType.ERROR)
        return response(code=Code.PROGRAM_ERR)


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    '''
    接口函数
    '''
    if request.method != 'POST':
        return response(code=Code.DATA_ERR)

    return_img = True if request.json.get('display_image') and request.json.get('display_image') == 'true' else False

    if request.files.get('image'):
        log('' + str(datetime.datetime.now()), LogType.INFO)

        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        log('read image ok' + str(datetime.datetime.now()), LogType.INFO)
        return predict_process(im, return_img)
        
    elif request.json.get('image_url'):
        img_url = request.json.get('image_url')
        p = urlparse(img_url)
        full_file_name = p.path.rsplit("/", 1)[-1]
        os.makedirs(TRMP_DIR, exist_ok=True)
        file_path = TRMP_DIR + str(datetime.datetime.now().timestamp()) + full_file_name
        print(img_url + "," + file_path)
        torch.hub.download_url_to_file(img_url, file_path)  # download 2 images
        im = Image.open(file_path)  # PIL image
        return predict_process(im, return_img)

    else:
        return response(code=Code.DATA_ERR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    log("device: cuda" if torch.cuda.is_available() else "device: cpu", LogType.WARN)         
    log('loading local models', LogType.WARN)
    models['scan'] = torch.hub.load(os.getcwd(), 'custom', source='local', path = 'D:/usr/yolov5s_gridDetection_best.pt', force_reload = True)
    # models['scan'] = torch.hub.load(os.getcwd(), 'custom', path_or_model='/usr/yolov5s_gridDetection_best.pt', source='local', force_reload = True)

    log('fusing mode layers to cuda', LogType.WARN)
    models['scan'].cuda()
    app.run(host='0.0.0.0', port=opt.port, debug=True)  # debug=True causes Restarting with stat
