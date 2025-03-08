import argparse
import torch
from decord import VideoReader, cpu
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
from PIL import Image
from transformers import AutoProcessor, AutoModel
from bson.binary import Binary
from bson.binary import BinaryVectorDtype
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex").to(device)
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

def extract_video_frames_in_batches(video_path, frame_skip_interval=2, batch_size=8):
    # Create a directory to save the images
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join("data", f"{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    video_reader = VideoReader(video_path, ctx=cpu(0))
    video_fps = video_reader.get_avg_fps()
    print("Video FPS:", video_fps)
    print("Frame Skip Interval:", frame_skip_interval)

    frame_batch = []
    frame_indexes = []
    seek_times = []
    frame_paths = []

    selected_frame_indexes = list(range(0, len(video_reader), frame_skip_interval + 1))

    for frame_index in selected_frame_indexes:
        frame = video_reader[frame_index].asnumpy()
        frame_batch.append(frame)
        frame_indexes.append(frame_index)
        
        seek_time = round(frame_index / video_fps, 2)
        seek_times.append(seek_time)

        # Save each frame in the batch as an image file
        image = Image.fromarray(frame)
        image_path = os.path.join(save_dir, f"{timestamp}_frame_{frame_index}.jpg")
        image.save(image_path, quality=85)  # Adjust quality (85 is a good balance)
        frame_paths.append(image_path)

        if len(frame_batch) == batch_size:
            yield np.array(frame_batch), frame_indexes, seek_times, frame_paths
            frame_batch = []
            frame_indexes = []
            seek_times = []
            frame_paths = []
    
    if frame_batch:
        yield np.array(frame_batch), frame_indexes, seek_times, frame_paths

def generate_bson_vector(vector, vector_dtype):
    return Binary.from_vector(vector, vector_dtype)

def main(video_file_path, frame_skip_interval, batch_size):
    client = MongoClient(MONGODB_URI)
    db = client['video_analysis']
    frames_collection = db['frames']

    frame_generator = extract_video_frames_in_batches(video_file_path, frame_skip_interval, batch_size)
    total_frames = len(VideoReader(video_file_path, ctx=cpu(0)))
    selected_frame_indexes = list(range(0, total_frames, frame_skip_interval + 1))

    for frame_batch, frame_indexes, seek_times, frame_paths in tqdm(frame_generator, total=len(selected_frame_indexes) // batch_size):
        inputs = processor(images=frame_batch, return_tensors="pt").to(device)

        with torch.no_grad():
            float32_embeddings = model.get_image_features(**inputs)

        float32_embeddings = float32_embeddings.cpu().numpy().astype(np.float32)

        bson_float32_embeddings = []
        for f32_emb in float32_embeddings:
            bson_float32_embeddings.append(generate_bson_vector(f32_emb, BinaryVectorDtype.FLOAT32))

        docs = []
        for frame_index, seek_time, embedding, frame_path in zip(frame_indexes, seek_times, bson_float32_embeddings, frame_paths):
            doc = {
                "frame_index": frame_index,
                "frame_path": frame_path,
                "seek_time": seek_time,
                "embedding": embedding,
            }
            docs.append(doc)

        frames_collection.insert_many(docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames and store embeddings in MongoDB.")
    parser.add_argument("--video_file_path", type=str, help="Path to the video file.")
    parser.add_argument("--frame_skip_interval", type=int, default=2, help="Number of frames to skip between extractions.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of frames per batch.")
    
    args = parser.parse_args()

    main(args.video_file_path, args.frame_skip_interval, args.batch_size)
