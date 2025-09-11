"""
This script is used to prepare the Cambrian737k dataset in the webdataset format.

Adopted from scripts/vlm/convert_to_qwen2vl_wds.py

"""

import argparse
import io
import json
import os
import pickle
import queue
import tarfile
import threading
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers

os.environ["FORCE_QWENVL_VIDEO_READER"] = 'torchvision'
from qwen_vl_utils import fetch_image, fetch_video


def fetch_image(img_path: str | Path) -> bytes:
    with Image.open(img_path) as image:
        with io.BytesIO() as buf:
            image.convert("RGB").save(buf, format="JPEG")
            # getvalue() returns a copy of the buffer content
            image_data = buf.getvalue()
    return image_data


def process_single_sample(indexed_entry, dataset_dir):
    """
    Process a single image entry and return the sample data.

    Args:
        indexed_entry: Tuple of (idx, entry) where idx is index and entry is dictionary containing image path and conversations
        dataset_dir: Path to the data directory

    Returns:
        Dictionary with processed sample data or None if image doesn't exist
    """

    idx, entry = indexed_entry

    # NOTE: read a dataset in sharegpt format
    images_data = []
    if 'image' in entry:
        pop_item = entry.pop('image')
    elif 'images' in entry:
        pop_item = entry.pop('images')
    else:
        pop_item = []

    if not isinstance(pop_item, list):
        pop_item = [pop_item]
    for image in pop_item:
        file_path = (dataset_dir / image).resolve()
        try:
            # NOTE:
            #   Due to limited disk space, used custom fetch_image function
            #   that skips image processing. This may impact training accuracy,
            #   so we need to carefully monitor the results.
            # fimage = fetch_image({"image": str(file_path)})
            # image_data = imageencoder(fimage)
            image_data = fetch_image(file_path)
            images_data.append(image_data)
        except Exception as e:
            print(f"ERROR: Failed to load image {file_path} for sample {idx}: {e}")
            raise

    videos_data = []
    if 'video' in entry:
        pop_item = entry.pop('video')
    elif 'videos' in entry:
        pop_item = entry.pop('videos')
    else:
        pop_item = []

    if not isinstance(pop_item, list):
        pop_item = [pop_item]
    for video in pop_item:
        file_path = (dataset_dir / video).resolve()
        fvideo = fetch_video({"video": str(file_path)})
        videos_data.append(fvideo)

    if 'conversations' in entry:
        conv = json.dumps(entry['conversations']).encode("utf-8")
    elif 'messages' in entry:
        conv = json.dumps(entry['messages']).encode("utf-8")
    else:
        conv = None

    if conv is None:
        print(f"ERROR: No conversation texts found for sample {idx}")
        print(f"Available keys in entry: {list(entry.keys())}")
        raise ValueError(f"No conversation texts found for sample {idx}")

    sample = {
        "__key__": entry.pop('id', str(idx)),
        "jpgs": images_data,  # Image bytes. SharedWrite can directly pickle them.
        'videos': videos_data,
        "json": conv,
    }
    return sample


def free_memory(sample: dict):
    for img in sample.pop('jpgs', []):
        del img
    for video in sample.pop('videos', []):
        del video


def filter_cambrian737k_dataset(
    dataset_dir: Path,
    metadata_json_path: Path,
    filtered_json_path: Path,
):
    """
    Filter the Cambrian737k dataset by checking if the image exists.

    Args:
        dataset_dir: Path to the data directory
        metadata_json_path: Path to the metadata JSON file
        filtered_json_path: Path to the filtered JSON file
    """

    with metadata_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # A snapshot of Cambrian dataset contains tar files.
    # Iterate over all files and folders in the image_root directory.
    for tar_path in dataset_dir.glob('*.tar'):
        print(f'Extracting {tar_path}...')
        expected_dir_name = tar_path.stem
        expected_dir_path = dataset_dir / expected_dir_name

        if expected_dir_path.exists():
            print(f"Directory '{expected_dir_name}' already exists. Skipping extraction.")
            continue

        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=dataset_dir)
            print(f'Successfully extracted {tar_path}.')
        except Exception as e:
            print(f'Error extracting {tar_path}: {e}')

    result = []

    for item in tqdm(data):
        image_path = item.get("image")
        if image_path is not None:
            full_path = (dataset_dir / image_path).resolve()
            if full_path.exists():
                result.append({
                    "image": image_path,
                    "conversations": item.get("conversations")
                })
            else:
                print(f"Image {image_path} does not exist.")
    print(f"{len(result)} conversations will be saced")

    with filtered_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Filtering done and saved to {filtered_json_path}.")


def convert_cambrian737k_dataset_to_webdataset(
    data_dir: Path,
    metadata_json_path: Path,
    output_dir: Path,
    num_workers: int | None = None,
):
    """
    Convert the Cambrian737k dataset to the webdataset format using threading for read/write separation.

    Args:
        data_dir: Path to the data directory
        metadata_json_path: Path to the metadata JSON file
        output_dir: Path to the output directory
        num_workers: Number of worker processes (default: min(32, CPU count // 2))
    """
    if num_workers is None:
        # Use half of CPU cores for I/O bound tasks like image processing
        num_workers = max(1, cpu_count() // 2)

    # Load data
    with metadata_json_path.open('r') as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a partial function with dataset_dir fixed
    process_func = partial(process_single_sample, dataset_dir=data_dir)

    print(f"Processing {len(data)} images using {num_workers} workers with threading...")

    # custom webdataset ShardWriter Encoder
    # Each image is already encoded in JPEG format. So we can directly pickle them.
    add_handlers(
        default_handlers, "jpgs", lambda data: pickle.dumps(data))
    # add_handlers(
    #     default_handlers, "jpgs", lambda data: pickle.dumps([np.array(d) for d in data]))
    add_handlers(
        default_handlers, "videos", lambda data: pickle.dumps([[np.array(d) for d in video] for video in data]))

    # Queue for communication between read and write threads
    batch_size = num_workers
    # Limit queue size to prevent memory overflow
    sample_queue = queue.Queue(maxsize=max(100, 4 * batch_size))

    def read_worker():
        """Read and process samples in background thread"""
        try:
            with tqdm(total=len(data), desc="Reading samples", position=0, leave=True) as pbar:
                with Pool(processes=num_workers) as pool:
                    for batch_start in range(0, len(data), batch_size):
                        batch_end = min(batch_start + batch_size, len(data))
                        batch_data = data[batch_start:batch_end]

                        batch_results = pool.map(process_func, enumerate(batch_data, batch_start))

                        # Put results in queue - block until space is available
                        for sample in batch_results:
                            if sample is not None:
                                sample_queue.put(sample)  # Block until space available
                                pbar.update(1)

            print("Read worker finished processing all batches")
        except Exception as e:
            print(f"Error in read worker: {e}")
            # Don't re-raise the exception to allow graceful shutdown
        finally:
            print("Read worker finished processing all batches, sending end signal to write worker")
            sample_queue.put(None)  # End signal for write worker

    def write_worker():
        """Write samples to shard writer in background thread"""
        try:
            with wds.ShardWriter(
                str(output_dir / 'Cambrian737k-%05d.tar'), maxcount=10000
            ) as shard_writer:
                with tqdm(total=len(data), desc="Writing samples", position=1, leave=True) as pbar:
                    while True:
                        sample = sample_queue.get()
                        if sample is None:  # End signal
                            print("End signal received, stopping write worker")
                            break
                        shard_writer.write(sample)
                        free_memory(sample)
                        sample_queue.task_done()
                        pbar.update(1)

            print("Write worker finished writing all samples")
        except Exception as e:
            print(f"Error in write worker: {e}")
            raise

    # Start read and write threads
    read_thread = threading.Thread(target=read_worker, name="ReadWorker")
    write_thread = threading.Thread(target=write_worker, name="WriteWorker")

    print("Starting read and write threads...")
    read_thread.start()
    write_thread.start()

    # Wait for both threads to complete
    read_thread.join()
    write_thread.join()

    print("Dataset successfully converted to the webdataset format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the Cambrian737k dataset in the webdataset format.")
    parser.add_argument(
        "--data-dir", type=Path,
        default='/datasets/Cambrian737k/Cambrian737k',
        help="Path to dataset directory.")
    parser.add_argument(
        "--output-dir", type=Path,
        default='/datasets/wds',
        help="Path to the output directory")
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Force to re-run data preparation.")
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of worker processes for parallel processing (default: max(1, CPU count // 2))")
    args = parser.parse_args()

    original_metadata_path = args.data_dir / 'Cambrian737k.json'
    filtered_metadata_path = args.data_dir / 'Cambrian737k_filtered.json'

    # Filter the dataset by checking if the image exists.
    if args.force or not filtered_metadata_path.exists():
        filter_cambrian737k_dataset(
            args.data_dir,
            original_metadata_path,
            filtered_metadata_path)
    else:
        print(f"Filtered metadata already exists at {filtered_metadata_path}. Skipping filtering.")

    if args.force or not args.output_dir.exists():
        convert_cambrian737k_dataset_to_webdataset(
            args.data_dir,
            filtered_metadata_path,
            args.output_dir,
            args.num_workers)
    else:
        print(f"Webdataset already exists at {args.output_dir} . Skipping conversion.")
