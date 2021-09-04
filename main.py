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


import os
import uvicorn
import threading
import translator as nmt


SERVER_STOP = False


def main():
    port = int(os.environ.get('PORT', nmt.DEFAULT_PORT))
    uvicorn.run(nmt.app, host=nmt.DEFAULT_HOST, port=port)
    SERVER_STOP = True


if __name__ == '__main__':
    model = threading.Thread(target=nmt.load_model, args=())
    server = threading.Thread(target=main, args=())
    while not SERVER_STOP: pass

