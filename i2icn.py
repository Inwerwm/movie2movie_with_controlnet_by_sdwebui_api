import json
from typing import Tuple
import numpy as np
import cv2
import methods

def load_image(path: str) -> Tuple[np.ndarray, int, int]:
    with open(path, 'rb') as f:
        img = f.read()

    img_arr = cv2.imdecode(np.frombuffer(img, dtype='uint8'), cv2.IMREAD_COLOR)
    height, width, _ = img_arr.shape

    return img_arr, height, width

with open("input_img.json", "r", encoding='utf-8') as f:
    input_params = json.load(f)

# 画像を読み込む
images, heights, widths = zip(*[load_image(path) for path in [input_params["input_color_path"], input_params["input_depth_path"], input_params["input_pose_path"]]])

# 画像をBase64形式にエンコード
image_base64s = [methods.image_to_base64(image, 1920) for image in images] # type: ignore

# Web API に送信
image = methods.img_to_img(image_base64s, widths[0], heights[0], input_params) # type: ignore

# 結果の画像を動画ファイルに書き込む
cv2.imwrite(input_params["output_path"], image)
