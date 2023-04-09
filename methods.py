import base64
import os
import random
import cv2
from typing import Dict, Any
import numpy as np

import requests

def create_request_json(color_frame_base64: str, depth_frame_base64: str, pose_frame_base64: str, frame_width: int, frame_height: int, denoising_strength: float, prompt: str, negative_prompt: str = "", seed: int = -1) -> Dict[str, Any]:
    return {
        "init_images": [color_frame_base64],
        "resize_mode": 0,
        "denoising_strength": denoising_strength,
        "image_cfg_scale": 0,
        "mask": None,
        "mask_blur": 4,
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 0,
        "inpainting_mask_invert": 0,
        "prompt": prompt,
        "styles": [""],
        "seed": seed,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": frame_width,
        "height": frame_height,
        "restore_faces": False,
        "tiling": False,
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "negative_prompt": negative_prompt + ", bad anatomy, painted by bad-artist, EasyNegative, (worst quality, low quality:1.4)",
        "eta": 0,
        "s_churn": 0,
        "s_tmax": 0,
        "s_tmin": 0,
        "s_noise": 1,
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "script_args": [],
        "sampler_index": "k_dpmpp_2m_ka",
        "include_init_images": False,
        "script_name": "",
        "send_images": True,
        "save_images": False,
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "module": "none",
                        "model": "control_depth-fp16 [400750f6]",
                        "weight": 1.0,
                        "image": depth_frame_base64,
                        "resize_mode": "Scale to Fit (Inner Fit)",
                        "low_vram": False,
                        "processor_res": 64,
                        "threshold_a": 64,
                        "threshold_b": 64,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "guess_mode": True
                    },
                    {
                        "module": "none",
                        "model": "control_openpose-fp16 [9ca67cc5]",
                        "weight": 1.0,
                        "image": pose_frame_base64,
                        "resize_mode": "Scale to Fit (Inner Fit)",
                        "low_vram": False,
                        "processor_res": 64,
                        "threshold_a": 64,
                        "threshold_b": 64,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "guess_mode": True
                    }
                ]
            }
        }
    }

def resize_if_too_large(image, max_size: int = 1024):
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        h_new, w_new = int(height * scale), int(width * scale)
        return cv2.resize(image, (w_new, h_new))
    else:
        return image

def image_to_base64(image: np.ndarray, max_size: int = -1) -> str:
    """
    画像をBase64形式にエンコードする

    Parameters
    ----------
    image : np.ndarray
        エンコードする画像データ
    
    max_size : int
        最大画像サイズ
        長辺の長さがこの数値以上の画像だった場合にリサイズする
        0以下の値が設定された場合はリサイズしない

    Returns
    -------
    str
        エンコードされた画像のBase64文字列
    """

    if max_size > 0:
        image = resize_if_too_large(image, max_size)
    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

def image_from_response_json(response_json: Dict[str, Any]) -> np.ndarray:
    """
    API のレスポンス JSON から画像フレームを復元する

    Parameters
    ----------
    response_json : dict
        API のレスポンス JSON

    Returns
    -------
    np.ndarray
        結果の画像データ
    """

    if 'images' in response_json:
        # 200 の場合
        encoded_img = response_json['images'][0]
        decoded_img = base64.b64decode(encoded_img)
        img_array = np.frombuffer(decoded_img, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    else:
        # 422 の場合
        raise ValueError(response_json['detail'][0]['msg'])
    
def img_to_img(base64_images: list[str], height: int, width: int, params) -> np.ndarray:
    request_json = create_request_json(*base64_images, width, height, params["denoising_strength"], params["prompt"], params["negative_prompt"], -1 if params.get("seed") is None else params["seed"]) # type: ignore
    res = requests.post('http://127.0.0.1:7860/sdapi/v1/img2img', json = request_json)
    return image_from_response_json(res.json())

def randomDigits(digits):
    lower = 10**(digits-1)
    upper = 10**digits - 1
    return random.randint(lower, upper)

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False