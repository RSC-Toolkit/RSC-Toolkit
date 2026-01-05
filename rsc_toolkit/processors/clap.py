import torch
from typing import Optional, Dict, Any
import logging

from rsc_toolkit.processors.base import BaseAudioProcessor

logger = logging.getLogger(__name__)


class CLAPProcessor(BaseAudioProcessor):
    """
    CLAP (Contrastive Language-Audio Pre-training) feature extractor.
    This is expensive, so caching is highly recommended.
    
    Requires: laion_clap package
    Install with: pip install laion-clap
    """
    model_name: str = "laion/clap-htsat-unfused"
    device: str = "cuda"  # or "cpu"
    batch_size: int = 32  # For batch processing during pre-computation
    _model: Optional[torch.nn.Module] = None
    
    def _get_processor_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
        }
    
    @property
    def model(self):
        """Lazy load CLAP model"""
        if self._model is None:
            try:
                from laion_clap import CLAP_Module
            except ImportError:
                raise ImportError(
                    "CLAP processor requires laion_clap package. "
                    "Install with: pip install laion-clap"
                )
            
            self._model = CLAP_Module(
                amodel_name="HTSAT-base",
                tmodel_name="roberta",
                enable_fusion=False,
                device=self.device
            )
            self._model.load_ckpt(self.model_name)
            logger.info(f"Loaded CLAP model: {self.model_name} on {self.device}")
        return self._model
    
    def make_features(
        self, 
        audio: torch.Tensor, 
        sample_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract CLAP features from audio.
        Audio should be shape [channels, samples] or [samples]
        """
        model = self.model
        
        # Ensure audio is on correct device and format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        
        # Move to device
        audio = audio.to(self.device)
        
        # Extract features (this is the expensive operation)
        with torch.no_grad():
            features = model.get_audio_embedding_from_data(
                audio, use_tensor=True
            )
        
        return features.cpu()  # Return to CPU for caching
    
    def make_features_batch(
        self,
        audio_batch: torch.Tensor,
        sample_ids: Optional[list[str]] = None
    ) -> list[torch.Tensor]:
        """
        Extract CLAP features from a batch of audio samples.
        More efficient than processing individually.
        
        Args:
            audio_batch: Tensor of shape [batch, channels, samples] or list of tensors
            sample_ids: Optional list of sample IDs for caching
        
        Returns:
            List of feature tensors
        """
        model = self.model
        
        # Handle list of tensors
        if isinstance(audio_batch, list):
            # Pad to same length
            max_len = max(a.shape[-1] for a in audio_batch)
            batch_tensor = torch.zeros(
                len(audio_batch), 
                audio_batch[0].shape[0] if audio_batch[0].dim() > 1 else 1,
                max_len,
                device=self.device
            )
            for i, a in enumerate(audio_batch):
                if a.dim() == 1:
                    a = a.unsqueeze(0)
                batch_tensor[i, :, :a.shape[-1]] = a.to(self.device)
            audio_batch = batch_tensor
        else:
            audio_batch = audio_batch.to(self.device)
        
        # Extract features for batch
        with torch.no_grad():
            features = model.get_audio_embedding_from_data(
                audio_batch, use_tensor=True
            )
        
        # Return as list of CPU tensors
        return [f.cpu() for f in features]

