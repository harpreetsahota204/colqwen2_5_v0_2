import logging
import os
from packaging.version import Version
import warnings
import math
from PIL import Image

import numpy as np

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

import torch
import torch.nn.functional as F

from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5 as ColQwenModel, ColQwen2_5_Processor

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ColQwenConfig(fout.TorchImageModelConfig):
    """
    This config class extends TorchImageModelConfig to provide specific parameters
    needed for ColQwen used for visual document retrieval.
    
    ColQwen is a multi-vector retrieval model that generates ColBERT-style 
    representations for both images and text queries.
    
    Args:
        model_path (str): Path to the model's weights on disk or HuggingFace model ID.
        
        text_prompt (str): Optional baseline text prompt to use for classification.
            Defaults to "".
        
        pool_factor (int): Token pooling compression factor. Default is 3 (optimal).
            Higher values = more compression, lower accuracy.
        
        pooling_strategy (str): Final pooling strategy after token pooling.
            Options: "mean" (default) or "max".
            - "mean": Average pooling, good for holistic document matching
            - "max": Max pooling, good for specific content/keyword matching
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        # ColQwenProcessor handles all preprocessing, so we use raw inputs
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        
        # Only set up the output processor if classes are provided (for classification)
        # If no classes, this model will be used only for embeddings
        if "classes" in d and d["classes"] is not None and len(d["classes"]) > 0:
            if "output_processor_cls" not in d:
                d["output_processor_cls"] = "fiftyone.utils.torch.ClassifierOutputProcessor"
        
        # Let parent class handle everything, including classes
        super().__init__(d)
        
        # Path to model weights or HuggingFace model ID
        self.model_path = self.parse_string(d, "model_path", default="")
        
        # Optional base text prompt
        self.text_prompt = self.parse_string(d, "text_prompt", default="")
        
        # Token pooling configuration
        self.pool_factor = self.parse_int(d, "pool_factor", default=3)
        self.pooling_strategy = self.parse_string(
            d, "pooling_strategy", default="mean"
        )
        
        # Validate pooling strategy
        if self.pooling_strategy not in ["mean", "max"]:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'max', "
                f"got '{self.pooling_strategy}'"
            )


class ColQwen(fout.TorchImageModel, fom.PromptMixin):
    """
    This model leverages ColQwen, a Vision Language Model based on QwenVL2.5,
    to create multi-vector embeddings for both images and text in a shared vector space,
    enabling visual document retrieval.

    
    The model can:
    1. Embed images into multiple vectors
    2. Embed text queries into multiple vectors
    3. Calculate multi-vector similarity between images and text
    4. Support visual document retrieval
    
    It extends TorchImageModel for image processing capabilities and PromptMixin to
    enable text embedding capabilities.
    """
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A ColQwenConfig instance containing model parameters
        """
        # Initialize parent classes (this sets self._classes from config.classes)
        super().__init__(config)
        
        # Storage for text features and embeddings
        self._text_features = None  # Cached text features for classification
        self._last_computed_embeddings = None  # Last computed 1D image embeddings
        self._last_computed_multi_vector_embeddings = None  # Store token-pooled multi-vector embeddings
        
        # Initialize token pooler for compression
        self.token_pooler = HierarchicalTokenPooler()
        self.pool_factor = config.pool_factor
        self.pooling_strategy = config.pooling_strategy

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.
        
        Returns:
            bool: Always True for this model as embedding generation is supported
        """
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts.
        
        Returns:
            bool: Always True for this model as text embedding is supported
        """
        return True
    
    def _apply_final_pooling(self, embeddings):
        """Apply final pooling strategy to token-pooled embeddings.
        
        Reduces multi-vector embeddings to fixed-dimension vectors for FiftyOne compatibility.
        
        Args:
            embeddings: Token-pooled embeddings with shape (batch, reduced_vectors, dim)
            
        Returns:
            torch.Tensor: Fixed-dimension pooled embeddings with shape (batch, dim)
        """
        if self.pooling_strategy == "mean":
            # Average across all vectors
            pooled = embeddings.mean(dim=1)  # (batch, dim)
            return pooled
        elif self.pooling_strategy == "max":
            # Take maximum across all vectors
            pooled = embeddings.max(dim=1)[0]  # (batch, dim)
            return pooled
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

    def _load_model(self, config):
        """Load the model and processor from disk or HuggingFace.
        
        This method initializes both the processor (for tokenization and image
        preprocessing) and the model itself, configuring them for inference.

        Args:
            config: ColQwenConfig instance containing model parameters

        Returns:
            The loaded PyTorch model ready for inference
        """
        # Load the model from HuggingFace or local path
        model_path = config.model_path

        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self._device)
            
            # Use bfloat16 for Ampere or newer GPUs (capability >= 8.0)
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16

        # Enable flash attention if available
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Initialize processor
        self.processor = ColQwen2_5_Processor.from_pretrained(model_path)
        
        # Initialize model
        self.model = ColQwenModel.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        return self.model

    def _get_text_features(self):
        """Get or compute text features for the model's classification.
        
        This method caches the result for efficiency in repeated calls.
        Creates embeddings for each class by combining text_prompt with class names.
        
        Returns:
            torch.Tensor: Token-pooled multi-vector text features for classification
        """
        # Check if text features are already computed and cached
        if self._text_features is None:
            # Create prompts for each class (following CLIP pattern)
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            # Compute and cache the text features for all classes
            self._text_features = self._embed_prompts(prompts)
        
        # Return the cached features
        return self._text_features
    
    @property
    def classes(self):
        """The list of class labels for the model."""
        return self._classes

    @classes.setter
    def classes(self, value):
        """Set new classes and invalidate cached text features."""
        self._classes = value
        self._text_features = None
        
        if value is not None and len(value) > 0:
            # Import and instantiate directly
            from fiftyone.utils.torch import ClassifierOutputProcessor
            self._output_processor = ClassifierOutputProcessor(classes=value)
        else:
            self._output_processor = None
                
        @property
        def text_prompt(self):
            """The text prompt prefix for classification."""
            return self.config.text_prompt

        @text_prompt.setter  
        def text_prompt(self, value):
            """Set new text prompt and invalidate cached text features."""
            self.config.text_prompt = value
            self._text_features = None  # Invalidate cache

    def _embed_prompts(self, prompts):
        """Embed text prompts for classification.
        
        Uses ColQwen's multi-vector embedding approach with token pooling for compression.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            torch.Tensor: Token-pooled multi-vector embeddings with shape (batch, reduced_num_vectors, embedding_dim)
        """
        # Process text queries using ColQwen2_5_Processor
        batch_queries = self.processor.process_queries(prompts).to(self.device)
        
        # Get query embeddings (multi-vector)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        
        # Apply token pooling to compress sequence length
        pooled_embeddings = self.token_pooler.pool_embeddings(
            query_embeddings,
            pool_factor=self.pool_factor,
            padding=True,
            padding_side=self.processor.tokenizer.padding_side,
        )
        
        # Return token-pooled multi-vector embeddings for classification
        return pooled_embeddings

    def embed_prompt(self, prompt):
        """Embed a single text prompt to 1D vector for retrieval.
        
        Uses token pooling + final pooling to create a single vector.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: 1D embedding vector with shape (dim,)
        """
        # Get token-pooled multi-vector embeddings
        embeddings = self._embed_prompts([prompt])
        
        # Apply final pooling strategy to get 1D vector
        final_embeddings = self._apply_final_pooling(embeddings)
        
        # Return first (and only) embedding: (1, dim) -> (dim,)
        result = final_embeddings[0].detach().cpu().numpy()
        return result

    def embed_prompts(self, prompts):
        """Embed multiple text prompts to 1D vectors for retrieval.
        
        Uses token pooling + final pooling to create single vectors.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: 1D embeddings with shape (batch, dim)
        """
        # Get token-pooled multi-vector embeddings
        embeddings = self._embed_prompts(prompts)
        
        # Apply final pooling strategy to get 1D vectors
        final_embeddings = self._apply_final_pooling(embeddings)
        
        # Return as numpy array
        result = final_embeddings.detach().cpu().numpy()
        return result

    def embed_images(self, imgs):
        """Embed a batch of images.
        
        With raw_inputs=True, FiftyOne passes images in their original format
        (PIL, numpy array, or tensor). ColQwen2_5_Processor requires PIL Images.
        
        Returns 1D embeddings for retrieval, but also stores token-pooled
        multi-vector embeddings internally for classification.
        
        Args:
            imgs: List of images to embed (PIL images, numpy arrays (HWC), or tensors (CHW))
            
        Returns:
            numpy array: 1D embeddings with shape (batch, dim)
        """
        # Convert to PIL Images if needed (ColQwen2_5_Processor requirement)
        pil_images = []
        for img in imgs:
            if isinstance(img, Image.Image):
                # Already PIL Image
                pil_images.append(img)
            elif isinstance(img, torch.Tensor):
                # Raw tensor (CHW, uint8) → PIL Image
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = img_np.astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                # Numpy array (HWC, uint8) → PIL Image
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        # Process images using ColQwen2_5_Processor
        batch_images = self.processor.process_images(pil_images).to(self.device)
        
        # Get image embeddings (multi-vector)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        
        # Apply token pooling to compress sequence length
        pooled_embeddings = self.token_pooler.pool_embeddings(
            image_embeddings,
            pool_factor=self.pool_factor,
            padding=True,
            padding_side=self.processor.tokenizer.padding_side,
        )
        
        # Store token-pooled multi-vector embeddings for classification
        self._last_computed_multi_vector_embeddings = pooled_embeddings
        
        # Apply final pooling to get 1D vectors for retrieval
        final_embeddings = self._apply_final_pooling(pooled_embeddings)
        
        # Cache final embeddings for get_embeddings() method
        self._last_computed_embeddings = final_embeddings
        
        # Return as numpy array
        result = final_embeddings.detach().cpu().numpy()
        return result
    
    def embed(self, img):
        """Embed a single image.
        
        Implementation of TorchEmbeddingsMixin.embed() method.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: 1D embedding with shape (dim,)
        """
        # Convert single image to a list for batch processing
        imgs = [img]
        
        # Embed the single image using the batch method
        embeddings = self.embed_images(imgs)
        
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Implementation of TorchEmbeddingsMixin.embed_all() method.
        
        Args:
            imgs: List of images to embed (PIL images)
            
        Returns:
            numpy array: 1D embeddings with shape (batch, dim)
        """
        # Directly call embed_images which handles batch processing
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed 1D embeddings.
        
        Required override for TorchEmbeddingsMixin to provide embeddings
        in the expected format for FiftyOne.
        
        Returns:
            numpy array: The last computed 1D embeddings
            
        Raises:
            ValueError: If no embeddings have been computed yet
        """
        # Check if embeddings capability is enabled
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        # Check if embeddings have been computed
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
        
        # Return the stored embeddings as a CPU numpy array
        result = self._last_computed_embeddings.detach().cpu().numpy()
        return result

    def _get_class_logits(self, text_features, image_features):
        """Calculate multi-vector similarity scores between text and image features.
        
        Uses ColQwen's multi-vector scoring approach similar to ColBERT's MaxSim operation.
        Both inputs are token-pooled multi-vector embeddings.
        
        Args:
            text_features: Token-pooled multi-vector text embeddings (torch.Tensor) 
                          with shape (num_classes, reduced_num_vectors, dim)
            image_features: Token-pooled multi-vector image embeddings (torch.Tensor) 
                           with shape (num_images, reduced_num_vectors, dim)
            
        Returns:
            tuple: (logits_per_image, logits_per_text) following CLIP convention
                - logits_per_image: shape (num_images, num_classes)
                - logits_per_text: shape (num_classes, num_images)
        """
        with torch.no_grad():
            # Use ColQwen2_5_Processor's scoring method for multi-vector similarity
            # Returns shape (num_classes, num_images)
            logits_per_text = self.processor.score_multi_vector(
                text_features, 
                image_features, 
                device=self.device
            )
            
            # Transpose to get (num_images, num_classes) for FiftyOne
            logits_per_image = logits_per_text.t()
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run prediction on a batch of images.
        
        Used for zero-shot classification by comparing image embeddings
        to text embeddings of class names using multi-vector similarity.
        
        Both image and text embeddings are token-pooled multi-vectors,
        ensuring consistent representation.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            numpy array: Similarity scores (logits)
        """
        # Check if classification is supported
        if self.classes is None or len(self.classes) == 0:
            raise ValueError(
                "Cannot perform classification without classes. "
                "Load the model with classes: "
                "foz.load_zoo_model(..., classes=['class1', 'class2', ...])"
            )
        
        if self._output_processor is None:
            raise ValueError(
                "No output processor configured for classification. "
                "This should not happen if classes were provided correctly."
            )
        
        # Get image embeddings (stores token-pooled multi-vector embeddings internally)
        _ = self.embed_images(imgs)
        
        # Get token-pooled multi-vector image embeddings
        image_features = self._last_computed_multi_vector_embeddings
        
        # Get token-pooled multi-vector text embeddings for classes
        text_features = self._get_text_features()
        
        # Calculate multi-vector similarity (following CLIP pattern)
        output, _ = self._get_class_logits(text_features, image_features)
        
        # Get frame size for output processor
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].size()[-2:]
        elif hasattr(imgs[0], 'size'):  # PIL Image
            width, height = imgs[0].size
        else:
            height, width = imgs[0].shape[:2]  # numpy array
        
        frame_size = (width, height)
        
        if self.has_logits:
            self._output_processor.store_logits = self.store_logits
        
        return self._output_processor(
            output, 
            frame_size, 
            confidence_thresh=self.config.confidence_thresh
        )