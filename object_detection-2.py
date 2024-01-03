import http.server
import socketserver
import urllib.request
import collections
import tarfile
import time
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo
from PIL import Image
from io import BytesIO
from tkinter import filedialog

# ... (your existing imports)

# Fetch notebook_utils module
# urllib.request.urlretrieve(
#     url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
#     filename='notebook_utils.py'
# )

# notebook_utilsをutilsファイルからインポート（オフライン用）
from utils import notebook_utils as utils

base_model_dir = Path("model")
model_name = "ssdlite_mobilenet_v2"
archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"
downloaded_model_path = base_model_dir / archive_name

if not downloaded_model_path.exists():
    utils.download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)

tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"

if not tf_model_path.exists():
    with tarfile.open(downloaded_model_path) as file:
        file.extractall(base_model_dir)

precision = "FP16"
converted_model_path = Path("model") / f"{model_name}_{precision.lower()}.xml"
trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"

if not converted_model_path.exists():
    ov_model = mo.convert_model(
        tf_model_path,
        compress_to_fp16=(precision == 'FP16'),
        transformations_config=trans_config_path,
        tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config",
        reverse_input_channels=True
    )
    ov.save_model(ov_model, converted_model_path)
    del ov_model

core = ov.Core()

print("Available devices:", core.available_devices)

# Manually input the device
# device_input = input("Enter the device (or 'AUTO' for automatic selection): ")
# device_value = device_input.upper() if device_input.upper() in core.available_devices + ["AUTO"] else 'AUTO'
# ask devices
# messagebox.showinfo("You can use these devices", core.available_devices)
# device_input = simpledialog.askstring("Device Selection", "Enter the device (or 'AUTO' for automatic selection):")
# device_value = device_input.upper() if device_input.upper() in core.available_devices + ["AUTO"] else 'AUTO'
use_device = 'CPU'
device_value = use_device if use_device in core.available_devices + ["AUTO"] else 'AUTO'

model = core.read_model(model=converted_model_path)
compiled_model = core.compile_model(model=model, device_name=device_value)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
height, width = list(input_layer.shape)[1:3]

classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]

colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

def process_results(frame, results, thresh=0.6):
    h, w = frame.shape[:2]
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        labels.append(int(label))
        scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
    )

    if len(indices) == 0:
        return []

    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        color = tuple(map(int, colors[label]))
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)

        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame

def convert_frame_for_html(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    return image_pil

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head><title>Object Detection Results</title></head><body>')
            self.wfile.write(b'<h1>Object Detection Results</h1>')
            self.wfile.write(b'<video controls width="640" height="480" src="/video"></video>')
            self.wfile.write(b'</body></html>')
        elif self.path == "/video":
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            player = utils.VideoPlayer(source=video_file, flip=False, fps=30, skip_first_frames=0)
            player.start()
            try:
                while True:
                    frame = player.next()
                    if frame is None:
                        break

                    input_img = cv2.resize(
                        src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
                    )
                    input_img = input_img[np.newaxis, ...]

                    results = compiled_model([input_img])[output_layer]
                    boxes = process_results(frame=frame, results=results)
                    frame = draw_boxes(frame=frame, boxes=boxes)

                    image_pil = convert_frame_for_html(frame)
                    img_byte_array = BytesIO()
                    image_pil.save(img_byte_array, format='PNG')
                    img_data = img_byte_array.getvalue()
                    
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Content-length', len(img_data))
                    self.end_headers()
                    self.wfile.write(img_data)
                    self.wfile.write(b'\r\n')

                    time.sleep(0.03)  # Adjust delay as needed for smooth playback

            except Exception as e:
                print(f"Error: {e}")
            finally:
                player.stop()
                self.wfile.write(b'--frame--\r\n')

if __name__ == "__main__":
    video_file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
    port = 8000
    handler = MyRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving on port {port}")
        httpd.serve_forever()
