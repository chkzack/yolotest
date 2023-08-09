FROM ultralytics/yolov5:latest

RUN pip install Flask

# Copy resources
COPY yolov5s.pt /usr/yolov5s.pt
COPY restapi.py /usr/restapi.py

RUN echo "python3 /usr/restapi.py --port 5000" >> /usr/start.sh && chmod 755 /usr/start.sh

CMD /usr/start.sh
