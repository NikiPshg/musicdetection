import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoFeatureExtractor
from typing import List, Dict, Sized

class MusicDetectionDataset(Dataset):
    """
    Loads and preprocesses individual audio files.
    Preprocessing is kept minimal here; batch-level operations like padding
    are handled by the collate function.
    """
    def __init__(
        self,
        file_paths: List[str],
        target_sample_rate: int = 16000
    ):
        self.file_paths = file_paths
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Loads and preprocesses a single audio file.
        Returns a dictionary containing the waveform and its original file path.
        """
        file_path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(file_path)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return {"waveform": waveform.squeeze(), "file_path": file_path}

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy tensor and path to avoid crashing the loader
            return {"waveform": torch.zeros(1), "file_path": file_path}


class AudioCollate:
    """
    A collate function that takes a list of preprocessed waveforms,
    extracts features using the provided processor, and pads them into a batch.
    """
    def __init__(self, processor: AutoFeatureExtractor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of samples from the Dataset into a single batch.
        """
        # Filter out any samples that failed to load
        valid_samples = [item for item in batch if item['waveform'].numel() > 1]
        
        if not valid_samples:
            return {} # Return empty dict if batch is empty after filtering

        waveforms = [item['waveform'].numpy() for item in valid_samples]
        file_paths = [item['file_path'] for item in valid_samples]

        inputs = self.processor(
            waveforms,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        
        # Keep file paths for tracking if needed
        inputs['file_paths'] = file_paths
        
        return inputs