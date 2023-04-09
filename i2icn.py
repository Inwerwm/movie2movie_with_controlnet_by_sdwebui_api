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

input_color_path = "G:\\VideoSource\\ドラキュラ否定_調整用_color.png"
input_depth_path = "G:\\VideoSource\\ドラキュラ否定_調整用_depth.png"
input_pose_path = "G:\\VideoSource\\ドラキュラ否定_調整用_pose.png"
output_path = "G:\\VideoSource\\ドラキュラ否定_調整用.png"

prompt = "1girl, blonde long hair twin tail, breasts, cleavage, collarbone, black elbow gloves, medium breasts, red eyes, (concrete wall:1.1), gold trimed blue dress, (glaring:0.8)"
negative_prompt = "bricks"

# 画像を読み込む
images, heights, widths = zip(*[load_image(path) for path in [input_color_path, input_depth_path, input_pose_path]])

# 画像をBase64形式にエンコード
image_base64s = [methods.image_to_base64(image, 1920) for image in images] # type: ignore

# Web API に送信
image = methods.img_to_img(image_base64s, widths[0], heights[0], prompt, negative_prompt) # type: ignore

# 結果の画像を動画ファイルに書き込む
cv2.imwrite(output_path, image)
