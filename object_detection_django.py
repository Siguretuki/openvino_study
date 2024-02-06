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
from tkinter import filedialog
from PIL import Image, ImageTk

# Fetch notebook_utils module
#urllib.request.urlretrieve(
#    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
#    filename='notebook_utils.py'
#)

# notebook_utilsをutilsフォルダからインポート（オフライン用）
from utils import notebook_utils as utils

#モデルの設定
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

use_device = 'CPU'
device_value = use_device if use_device in core.available_devices + ["AUTO"] else 'AUTO'

model = core.read_model(model=converted_model_path)
compiled_model = core.compile_model(model=model, device_name=device_value)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
height, width = list(input_layer.shape)[1:3]

classes = [
    "background", "person", "bicycle","motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "blender", "car",  "book", "clock", "vase", "scissors",
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

def convert_frame_for_tkinter(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image=image)
    return image

def run_object_detection(input_video_path,output_video_path):
    #入出力の設定
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path,fourcc,fps,(frame_width,frame_height))

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            #推論の完了、ここに結果ページへの移動プログラムを追記予定
            print("Source ended")
            break
        #前処理
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        input_img = cv2.resize(
            src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
        )
        input_img = input_img[np.newaxis, ...]
        #推論の実行
        start_time = time.time()
        results = compiled_model([input_img])[output_layer]
        stop_time = time.time()
        boxes = process_results(frame=frame, results=results)

        frame = draw_boxes(frame=frame, boxes=boxes)
        processing_times = collections.deque()
        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()

        _, f_width = frame.shape[:2]
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(
            img=frame,
            text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=f_width / 1000,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        #結果の書き込み
        out.write(frame)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#使う場所によって要変更
output_video_path = "./uploads/output_video.mp4"
input_video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
run_object_detection(input_video_path,output_video_path)
