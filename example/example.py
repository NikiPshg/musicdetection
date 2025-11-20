import torch
import torch.multiprocessing as mp
from transformers import AutoFeatureExtractor
from safetensors import safe_open
import numpy as np

from src.musicdetection.audio_cache import create_audio_length_cache
from src.musicdetection.dataset import MusicDetectionDataset, AudioCollate
from src.musicdetection.core.model import WavLMForMusicDetection
from src.musicdetection.audio_sampler import LengthBasedBatchSampler

def inference_worker(
    gpu_id: int,
    audio_paths_chunk: list,
    model_config: dict
):
    """
    The worker function that runs on a single GPU.
    It creates its own dataset, dataloader, and model instance.

    Args:
        gpu_id (int): The GPU device ID to use.
        audio_paths_chunk (list): The subset of audio files for this worker to process.
        model_config (dict): A dictionary containing model configuration.
    """
    # device = torch.device("cpu")
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}]: Process started, assigned {len(audio_paths_chunk)} files.")
    
    audio_lengths = create_audio_length_cache(audio_paths_chunk, cache_file=model_config['CACHE_FILE'])
    
    processor = AutoFeatureExtractor.from_pretrained(model_config['BASE_MODEL_NAME'])
    
    dataset = MusicDetectionDataset(
        file_paths=audio_paths_chunk,
        target_sample_rate=processor.sampling_rate
    )

    sampler = LengthBasedBatchSampler(
        file_paths=audio_paths_chunk,
        audio_lengths=audio_lengths,
        batch_size=model_config['BATCH_SIZE_PER_GPU'],
        shuffle=False
    )

    collate_fn = AudioCollate(processor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=model_config['NUM_WORKERS_PER_GPU'],
        pin_memory=True
    )

    for batch in dataloader:
        print(batch['input_values'].shape, batch['attention_mask'].shape)

    model = WavLMForMusicDetection(base_model_name=model_config['BASE_MODEL_NAME'])
    
    with safe_open(model_config['CHECKPOINT_PATH'], framework="pt", device="cpu") as f:
        model.load_state_dict({key: f.get_tensor(key) for key in f.keys()})

    model.to(device)    
    predictions = model.predict_proba(dataloader)
    print(predictions)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    config = {
        'BASE_MODEL_NAME': 'microsoft/wavlm-base-plus',
        'CHECKPOINT_PATH': './models/music_detection.safetensors',
        'CACHE_FILE': './audio_length_cache.json',
        'BATCH_SIZE_PER_GPU': 32,
        'NUM_WORKERS_PER_GPU': 16 # CPU workers for each GPU's dataloader
    }
    
    AUDIO_PATHS = ['audio_path.wav']

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one CUDA-enabled GPU.")
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Spawning inference workers...")

    audio_path_chunks = np.array_split(AUDIO_PATHS, num_gpus)
    
    processes = []

    for gpu_id in range(num_gpus):
        if len(audio_path_chunks[gpu_id]) > 0:
            process = mp.Process(
                target=inference_worker,
                args=(gpu_id, audio_path_chunks[gpu_id].tolist(), config)
            )
            processes.append(process)
            process.start()

    for process in processes:
        process.join()
