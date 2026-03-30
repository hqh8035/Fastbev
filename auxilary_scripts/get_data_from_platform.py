import os
import argparse
from pathlib import Path
from tqdm import tqdm

from dp_data_common.perception.pilot_frame_client import PilotFrameClient

def get_frames_from_platform(database, collection, query):
    frame_client = PilotFrameClient()
    frames = frame_client.find(database, collection, query=query, stream=True)
    return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load frames from data platform")
    parser.add_argument("--database", type=str, default='db_dev', help="Name of the database")
    parser.add_argument("--collection", type=str, default='sixents_label', help="Name of the collection")
    parser.add_argument("--query", type=str, default='{"info.meta.collect_date": {"$gte": "20251203"}}', help="query string of mongodb")
    parser.add_argument("--dst_dir", type=str, default='extracted_data', help="dst data dir")

    args = parser.parse_args()
    
    left_camera_name = 'camera_left'
    right_camera_name = 'camera_right'
    
    left_image_dir = Path(args.dst_dir) / 'left'
    right_image_dir = Path(args.dst_dir) / 'right'
    
    left_image_dir.mkdir(parents=True, exist_ok=True)
    right_image_dir.mkdir(parents=True, exist_ok=True)
    
    frames = get_frames_from_platform(args.database, args.collection, eval(args.query))
    for frame in tqdm(frames):
        left_image_file = frame.get_camera(left_camera_name).frame[0].image
        right_image_file = frame.get_camera(right_camera_name).frame[0].image

        try:
            os.symlink(left_image_file, left_image_dir / Path(left_image_file).name)
            # bev 模型，右相机的图像不需要
            # os.symlink(right_image_file, right_image_dir / Path(right_image_file).name)
        except FileExistsError as e:
            print(f"File {left_image_file} or {right_image_file} already exists, skipping")
            continue
        except Exception as e:
            print(f"Error symlinking {left_image_file} or {right_image_file}: {str(e)}")
            break