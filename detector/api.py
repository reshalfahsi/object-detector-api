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


from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from detector.__about__ import __description__ as DESCRIPTION
from detector.model import Detector


app = FastAPI()
detector = Detector()


"""Sanity Check"""
@app.get("/")
async def index():
    return DESCRIPTION


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = detector.predict(file)
    return StreamingResponse(result, media_type="image/png")


app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


