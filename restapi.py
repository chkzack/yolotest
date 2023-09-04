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

app = Flask(__name__)
models = {}

DETECTION_URL = '/v1/object-detection/scan'

class Response:
    def __init__(self):
        self.objectId = ''     # 名称
        self.typeList = ["bj_bjmh", "bj_bjds"] # 识别类型
        self.imageUrlList = ["jpg图像"] # 图片解析格式
        self.results = []     # 结果列表

class Result:
    def __init__(self):
        self.type = ''  # 分析类型
        self.value = '' # 值
        self.code = '2002' # 正确:2000, 图像数据错误:2001, 算法分析失败:2002
        self.resultImageUrl = ''
        self.pos = []
        self.conf = 0.0
        self.desc = ''

class Shape:
    def __init__(self):
        self.areas = []

class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

class ResponseEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    response1 = Response()
    response1.objectId = 'scan'

    if request.method != 'POST':
        result1 = Result()
        result1.code = '2001' # 图像数据错误
        response1.results.append(result1)
        return json.dumps(response1, cls=ResponseEncoder)

    if request.files.get('image'):
        app.logger.warning('' + str(datetime.datetime.now()))
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        app.logger.warning('read image ok' + str(datetime.datetime.now()))

        # if model in models:
        #     results = models[model](im, size=640)  # reduce size=320 for faster inference
        #     return results.pandas().xyxy[0].to_json(orient='records')

        try:
            results = models['scan'](im, size=640)  # reduce size=320 for faster inference
        # if request.args.get('display_image'):
        #     img = cv2.cvtColor(numpy.asarray(im), cv2.COLOR_RGB2BGR)

        #     for box in results.xyxy[0]: 
        #         print(box)
        #         print("draw:")
        #         xB = int(box[2])
        #         xA = int(box[0])
        #         yB = int(box[3])
        #         yA = int(box[1])
        #         cv2.rectangle(img, (xA, yA), (xB, yB), (0, 0, 255), 2)

        #     saved_path = '/tmp/' + str(time.time()) + "_precdit.jpg"
        #     cv2.imwrite(saved_path, img)

        #     return send_file(saved_path, mimetype='image/jpeg')
        # else:
            # app.logger.warning(results.pandas())
            app.logger.error(results.pandas().xyxy[0])

            for pred in results.pandas().xyxy[0].values.tolist():
                # 识别矩形起始，结束点位
                shape1 = Shape()

                rect_point_start = Point()
                rect_point_start.x = int(pred[0]) # x1
                rect_point_start.y = int(pred[1]) # y1
                shape1.areas.append(rect_point_start)

                rect_point_end = Point()
                rect_point_end.x = int(pred[2]) # x2
                rect_point_end.y = int(pred[3]) # y2
                shape1.areas.append(rect_point_end)

                confidencel = pred[4] # confidencel
                type_class = pred[5] # class
                name = pred[6] # name
                
                result1 = Result()
                result1.type = type_class # 分析类型
                result1.value = '' # 值
                result1.code = '2000' # 正确
                result1.resultImageUrl = ''
                result1.pos.append(shape1)
                
                result1.conf = confidencel
                result1.desc = name

                response1.results.append(result1)
            app.logger.warning('scan image ok' + str(datetime.datetime.now()))
            return json.dumps(response1, cls=ResponseEncoder)
        except Exception:
            result1 = result()
            result1.code = '2002' # 算法程序错误
            response1.results.append(result1)
            return json.dumps(response1, cls=ResponseEncoder)
    # elif request.json.get('image_url'):
    #     # Method 3
    #     img_url = request.json.get('image_url')
    #     p = urlparse(img_url)
    #     full_file_name = p.path.rsplit("/", 1)[-1]
    #     file_path = '/tmp/images/' + full_file_name
    #     print(img_url + "," + file_path)
    #     torch.hub.download_url_to_file(img_url, file_path)  # download 2 images
    #     im = Image.open(file_path)  # PIL image
    #     # im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

    #     results = models['scan'](im, size=640)  # reduce size=320 for faster inference
    #     if request.json.get('display_image'):
    #         img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
    #         for box in results.xyxy[0]: 
    #             print(box)
    #             print("draw:")
    #             xB = int(box[2])
    #             xA = int(box[0])
    #             yB = int(box[3])
    #             yA = int(box[1])
    #             cv2.rectangle(img, (xA, yA), (xB, yB), (0, 0, 255), 2)

    #         file = file_path.split(".")
    #         saved_path = file[0] + "_precdit." + file[1]
    #         cv2.imwrite(saved_path, img)
           
    #         return send_file(saved_path, mimetype='image/jpeg')
    #     else:
    #         app.logger.warning(results.pandas())
    #         # return results.pandas().xyxy[0].to_json(orient='records')
    #         response = response()
    #         response.objectId = 'scan' 
            
    #         '''
    #         # 响应:
    #         [{'class': 1,
    #         'confidence': 0.850515306,
    #         'name': 'grid_meter',
    #         'xmax': 453.3086853027,
    #         'xmin': 248.521484375,
    #         'ymax': 289.7153320312,
    #         'ymin': 74.1867980957},
    #         {'class': 1,
    #         'confidence': 0.4257921278,
    #         'name': 'grid_meter',
    #         'xmax': 257.9888916016,
    #         'xmin': 169.6451416016,
    #         'ymax': 449.7769165039,
    #         'ymin': 425.6575317383}]
    #         '''
    #         for categories in results.pandas().xyxy:
    #             result = result()

    #             result.type = ''  # 分析类型
    #             result.value = '' # 值
    #             result.code = '2000' # 正确
    #             result.resultImageUrl = ''

    #             # 识别矩形起始，结束点位
    #             for pos in categories:
    #                 shape = shape()

    #                 rect_point_start = point()
    #                 rect_point_start.x = int(pos[0])
    #                 rect_point_start.y = int(pos[1])
    #                 shape.areas.append(rect_point_start)

    #                 rect_point_end = point()
    #                 rect_point_end.x = int(pos[2])
    #                 rect_point_end.y = int(pos[3])
    #                 shape.areas.append(rect_point_end)

    #                 result.pos.append(shape)
                
    #             result.conf = pos.confidence
    #             result.desc = pos.name

    #         response.results.append(result)
    #         return response.to_json(orient='records')
    else:
        result1 = Result()
        result1.code = '2001' # 图像数据错误
        response1.results.append(result1)
        return json.dumps(response1, cls=ResponseEncoder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    app.logger.warning("cuda" if torch.cuda.is_available() else "cpu")         
    # for m in opt.model:
    #     models[m] = torch.hub.load('ultralytics/yolov5', m, force_reload=True, skip_validation=True)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='/usr/yolov5s_gridDetection_best.pt', force_reload=True, skip_validation=True)
    app.logger.warning('load local models')
    models['scan'] = torch.hub.load(os.getcwd(), 'custom', source='local', path = 'D:/usr/yolov5s_gridDetection_best.pt', force_reload = True)
    # models['scan'] = torch.hub.load(os.getcwd(), 'custom', path_or_model='/usr/yolov5s_gridDetection_best.pt', source='local', force_reload = True)
    app.logger.warning('use cuda')
    models['scan'].cuda()
    # model = torch.jit.load('/usr/yolov5s_gridDetection_best.pt').eval().toGpu()
    app.run(host='0.0.0.0', port=opt.port, debug=True)  # debug=True causes Restarting with stat
