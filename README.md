# Object Detector API

Minimal implementation of object detection API hosted on [Heroku](https://wpir-dnjf-8439.herokuapp.com/).

## Deployment

Run this following command and the API will be up and running on the cloud.

```bash
git clone https://github.com/reshafahsi/object-detector-api.git
cd object-detector-api
heroku create <your-app-name>
heroku git:remote <your-app-name>
heroku stack:set container
git push heroku master
```

## Running on Local

Run this following command for local usage.

```bash
git clone https://github.com/reshafahsi/object-detector-api.git
cd object-detector-api
sudo pip3 install -r requirements.txt
sudo docker build -t "object-detector-api:v1.0" .
sudo docker run -ti --name "object-detector-api" "object-detector-api:v1.0"
```

## API Reference

### GET /

Request: `None`

Response: `"Object Detector on Cloud with Deep Learning"`

### POST /predict

Request: Image file

Response: Bytearray of image file

