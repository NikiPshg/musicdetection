# Music Detection

A high-performance music detection library based on WavLM, optimized for multi-GPU inference with maximum GPU utilization.

## Features

- 🎵 **Music Detection**: Detect music in audio files using a fine-tuned WavLM model
- 🚀 **Multi-GPU Support**: Efficient parallel inference across multiple GPUs
- ⚡ **Optimized Batching**: Length-based batch sampling to minimize padding and maximize GPU utilization
- 💾 **Audio Cache**: Cache audio length information for faster batch processing
- 🔧 **Easy to Use**: Simple API for inference on single or multiple audio files

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/NikiPshg/musicdetection
```

## Quick Start

### Single GPU Inference

```python
import torch
from transformers import AutoFeatureExtractor
from safetensors import safe_open
from musicdetection.audio_cache import create_audio_length_cache
from musicdetection.dataset import MusicDetectionDataset, AudioCollate
from musicdetection.core.model import WavLMForMusicDetection
from musicdetection.audio_sampler import LengthBasedBatchSampler
from torch.utils.data import DataLoader

# Prepare audio files
audio_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']

# Create audio length cache for efficient batching
audio_lengths = create_audio_length_cache(
    file_paths=audio_paths,
    cache_file='audio_length_cache.json'
)

# Initialize processor
processor = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')

# Create dataset
dataset = MusicDetectionDataset(
    file_paths=audio_paths,
    target_sample_rate=processor.sampling_rate
)

# Create length-based batch sampler
sampler = LengthBasedBatchSampler(
    file_paths=audio_paths,
    audio_lengths=audio_lengths,
    batch_size=32,
    shuffle=False
)

# Create dataloader
collate_fn = AudioCollate(processor)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# Initialize model (base WavLM model will be loaded from HuggingFace)
model = WavLMForMusicDetection(base_model_name='microsoft/wavlm-base-plus')

# Load fine-tuned weights from checkpoint
from safetensors import safe_open
with safe_open('./models/music_detection.safetensors', framework="pt", device="cpu") as f:
    model.load_state_dict({key: f.get_tensor(key) for key in f.keys()})

model.to('cuda:0')
model.eval()

# Run inference
predictions = model.predict_proba(dataloader)
print(predictions)  # Tensor of probabilities [path1, path2], [0, 1] for each audio file
```

### Multi-GPU Inference

For multi-GPU inference, see the complete example in `example/example.py`. The example demonstrates:
- Splitting audio files across multiple GPUs
- Parallel processing with multiprocessing
- Efficient batch processing with length-based sampling

## Model Details

The model is based on **WavLM-base-plus** fine-tuned for music detection. It uses:
- **Attention-based pooling** to aggregate temporal features
- **Classification head** to output music probability (0-1)

### Model Weights

The model weights can be downloaded from HuggingFace Hub:
- Repository: `MTUCI/MusicDetection`
- File: `music_detection.safetensors`

**Note**: 
- The base WavLM model is automatically downloaded from HuggingFace when initializing `WavLMForMusicDetection`
- The fine-tuned weights (`music_detection.safetensors`) should be placed in `./models/` directory
- The model will attempt to automatically download fine-tuned weights from HuggingFace Hub (`MTUCI/MusicDetection`) if not found locally
- Make sure you have `huggingface-hub` installed and proper authentication if the repository requires it

## API Reference

### `MusicDetectionDataset`

Dataset class for loading and preprocessing audio files.

**Parameters:**
- `file_paths` (List[str]): List of paths to audio files
- `target_sample_rate` (int): Target sample rate (default: 16000)

### `AudioCollate`

Collate function for batching preprocessed waveforms.

**Parameters:**
- `processor` (AutoFeatureExtractor): HuggingFace feature extractor

### `LengthBasedBatchSampler`

Batch sampler that groups audio files by length to minimize padding.

**Parameters:**
- `file_paths` (List[str]): List of audio file paths
- `audio_lengths` (Dict[str, float]): Dictionary mapping file paths to durations
- `batch_size` (int): Batch size (default: 32)
- `shuffle` (bool): Whether to shuffle batches (default: True)

### `WavLMForMusicDetection`

Music detection model.

**Parameters:**
- `base_model_name` (str): HuggingFace model name (default: 'microsoft/wavlm-base-plus')

**Methods:**
- `forward(input_values, attention_mask)`: Forward pass
- `predict_proba(dataloader)`: Predict music probability for all files in dataloader

### `create_audio_length_cache`

Create or load cache of audio file lengths.

**Parameters:**
- `file_paths` (List[str]): List of audio file paths
- `cache_file` (str, optional): Path to cache file
- `num_workers` (int, optional): Number of workers for parallel processing
- `force_rebuild` (bool): Force rebuild cache (default: False)

**Returns:**
- `Dict[str, float]`: Dictionary mapping file paths to durations

## Performance Optimization

### Batch Size Tuning

- Use larger batch sizes (32-64) for maximum GPU utilization
- Adjust based on GPU memory and audio file lengths

### Multi-GPU Setup

- Split audio files evenly across available GPUs
- Use `num_workers=16` per GPU for optimal data loading
- Enable `pin_memory=True` for faster CPU-to-GPU transfer

### Audio Cache

- Cache audio lengths to avoid re-measuring files
- Speeds up batch sampling significantly for repeated inference

## License

MIT License

## Citation

If you use this library in your research, please cite:

```bibtex
@software{musicdetection2025,
  title={Music Detection: High-performance music detection using WavLM},
  author={NikiPshg},
  year={2025},
  url={https://github.com/NikiPshg/musicdetection}
}
```

## Acknowledgments

- Based on [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) by Microsoft
- Model weights from [MTUCI/MusicDetection](https://huggingface.co/MTUCI/MusicDetection)

