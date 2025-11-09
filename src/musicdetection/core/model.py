import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
from huggingface_hub import hf_hub_download

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

class WavLMForMusicDetection(nn.Module):
    """
    Music detection model based on WavLM.
    This class now focuses only on the model architecture and forward pass.
    Data loading and preprocessing are handled by the DataLoader.
    """
    def __init__(self, base_model_name: str = 'microsoft/wavlm-base-plus'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.wavlm = AutoModel.from_pretrained(base_model_name, config=self.config)
        self.device = next(self.parameters()).device
        if not os.path.exists('./models/music_detection.safetensors'):
            self._load_from_hf()

        # Attention-based pooling head
        self.pool_attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def _attention_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Applies attention-based pooling over the time dimension."""
        attention_weights = self.pool_attention(hidden_states)
        attention_weights.masked_fill_(~attention_mask.unsqueeze(-1), -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)
        return weighted_sum

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for inference.
        Args:
            input_values (torch.Tensor): [batch_size, audio_seq_len] — raw audio waveform
            attention_mask (torch.Tensor): [batch_size, audio_seq_len] — input mask (1 = real, 0 = pad)
        Returns:
            torch.Tensor: [batch_size, 1] — probability that audio contains music
        """
        assert isinstance(input_values, torch.Tensor), f"Expected torch.Tensor, got {type(input_values)}"
        assert isinstance(attention_mask, torch.Tensor), f"Expected torch.Tensor, got {type(attention_mask)}"

        outputs = self.wavlm(input_values.to(self.device), attention_mask=attention_mask.to(self.device))
        hidden_states = outputs.last_hidden_state  # [B, T', D]

        # Align attention mask with downsampled hidden states
        input_length = attention_mask.size(1)
        hidden_length = hidden_states.size(1)
        ratio = input_length / hidden_length
        indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
        attention_mask = attention_mask[:, indices]  # [B, T']
        attention_mask = attention_mask.bool()

        pooled = self._attention_pool(hidden_states, attention_mask) 
        logits = self.classifier(pooled)  # [B, 1]

        probs = torch.sigmoid(logits)  # [B, 1] → probability of MUSIC
        return probs

    @torch.inference_mode()
    def predict_proba(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Predicts music probability for all data in the dataloader.
        Args:
            dataloader (DataLoader): A DataLoader instance that yields batches of preprocessed data.
        Returns:
            torch.Tensor: A 1D tensor of probabilities for each audio file.
        """
        self.eval()
        
        all_probs = []

        for batch in tqdm(dataloader, desc="Predicting"):
            if not batch: # Skip empty batches
                continue
            
            # Move batch to the same device as the model
            # DataParallel will handle splitting it across GPUs
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                probs = self.forward(input_values=input_values, attention_mask=attention_mask).squeeze(-1)
            all_probs.append(probs)
            print(probs)
            print(batch['file_paths'])
            print("--------------------------------")

        return torch.cat(all_probs, dim=0)
    
    def _load_from_hf(self):
        return hf_hub_download(
            repo_id="MTUCI/MusicDetection",
            filename="music_detection.safetensors",
            local_dir="./models",         
            local_dir_use_symlinks=False  
        )