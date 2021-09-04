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


import io
import cv2
import torchvision

from PIL import Image
from detector.utils import (
    preprocessing,
    postprocessing
)


class Detector:
    def __init__(self):
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        self.model.eval()

    def predict(self, file):
        image = Image.open(file.file)
        tensor = preprocessing(image)

        prediction = self.model(tensor)

        result = postprocessing(image, prediction)

        response, result = cv2.imencode(".png", result)

        return io.BytesIO(result.tobytes())
