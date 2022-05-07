# MIT License
# 
# Copyright (c) 2021 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import cv2
import random
import numpy as np

from torchvision import transforms


DEFAULT_PORT = 80
DEFAULT_HOST = '0.0.0.0'


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


DETECTION_THRESHOLD = 0.5


def preprocessing(image):
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    return tensor


def postprocessing(image, prediction):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = prediction[0]
    
    for index in range(prediction['labels'].shape[0]):
        boxes = prediction['boxes'][index].detach().numpy()
        scores = prediction['scores'][index].detach().numpy()
        labels = COCO_INSTANCE_CATEGORY_NAMES[prediction['labels'][index].detach().numpy()]

        if scores > DETECTION_THRESHOLD:
            text = "{} - {}%".format(labels, round(scores*100., 3))
            start_point = (int(boxes[0]), int(boxes[1]))
            pos_txt = (int(boxes[0] + 10), int(boxes[1] + 40))
            end_point = (int(boxes[2]), int(boxes[3]))
            color = tuple(int(c) for c in tuple(np.random.choice(range(256), size=3)))
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            line = cv2.LINE_AA
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image = cv2.putText(image, text, pos_txt, font, fontScale, color, thickness, line)
    return image
