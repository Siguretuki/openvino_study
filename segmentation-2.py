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
import matplotlib.pyplot as plt

# notebook_utilsをutilsフォルダからインポート（オフライン用）
from utils import notebook_utils as notebook_utils
from utils.notebook_utils import download_file ,segmentation_map_to_image

# モデルの設定

base_model_dir = Path("model")
model_name = "road-segmentation-adas-0001"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'

model_xml_path = base_model_dir / model_xml_name

if not model_xml_path.exists():
    model_xml_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP32/{model_name}.xml"
    model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP32/{model_name}.bin"

    download_file(model_xml_url, model_xml_name, base_model_dir)
    download_file(model_bin_url, model_bin_name, base_model_dir)
else:
    print(f'{model_name} already downloaded to {base_model_dir}')

core = ov.Core()

use_device = 'CPU'
device_value = use_device if use_device in core.available_devices + ["AUTO"] else 'AUTO'

model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device_value)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
height, width = list(input_layer.shape)[1:3]

# たぶんこの辺が処理
def run_segmentation(input_video_path, output_video_path):
    # 入出力の設定
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 推論の完了、ここに結果ページへの移動プログラムを追記予定
            print("Source ended")
            break
        frame_count += 1
        # Load the image
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # N, C, H, W = batch size, number of channels, height, width.
        N, C, H, W = input_layer.shape

        # OpenCV resize expects the destination size as (width, height).
        resized_image = cv2.resize(frame, (W, H))

        # Reshape to the network input shape.
        input_img = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

        # Run inference
        start_time = time.time()
        result = compiled_model([input_img])[output_layer]
        stop_time = time.time()

        # Post-processing
        segmentation_mask = np.argmax(result, axis=1)
        colormap = np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 0, 0]])
        alpha = 0.5
        mask = segmentation_map_to_image(segmentation_mask, colormap)
        resized_mask = cv2.resize(mask, (frame_width, frame_height))
        image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)

        processing_time = (stop_time - start_time) * 1000
        cv2.putText(
            img=image_with_mask,
            text=f"Inference time: {processing_time:.1f}ms",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame_width / 1000,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Write the result
        out.write(image_with_mask)
        progress = frame_count / total_frames
        print(f"Progress: {progress * 100:.2f}%")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 使う場所によって要変更
output_video_path_segmentation = "./uploads/output_video_segmentation.mp4"
input_video_path_segmentation = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
run_segmentation(input_video_path_segmentation, output_video_path_segmentation)
