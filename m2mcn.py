import json
import cv2
from tqdm import tqdm
import methods

# 入力動画ファイル名と出力動画ファイル名
with open("input.json", "r", encoding='utf-8') as f:
    input_params = json.load(f)

# 入力動画を読み込む
caps = [cv2.VideoCapture(path) for path in [input_params["input_color_path"], input_params["input_depth_path"], input_params["input_pose_path"]]]
fps = caps[0].get(cv2.CAP_PROP_FPS)
frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

# シード値を生成
input_params["seed"] = methods.randomDigits(10)

# 出力動画を設定
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(input_params["output_path"], fourcc, fps, (frame_width, frame_height))

for _ in tqdm(range(total_frames)):
    # フレームを読み込む
    isSuccess, frames = zip(*[cap.read() for cap in caps])

    # どれか一つでも動画が終了していればループを抜ける
    if not all(isSuccess):
        break

    # フレーム画像をBase64形式にエンコード
    frame_base64s = [methods.image_to_base64(frame) for frame in frames]

    # Web API に送信
    result_frame = methods.img_to_img(frame_base64s, frame_height, frame_width, input_params)

    # 結果の画像を動画ファイルに書き込む
    out.write(result_frame)

# 動画ファイルを閉じる
for cap in caps:
    cap.release()
out.release()
