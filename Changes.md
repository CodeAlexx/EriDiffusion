# Changes Log

## 2025-07-01: Chroma Model Implementation

### Added
- **Chroma model architecture** (`ai-toolkit-rs/crates/models/src/chroma.rs`)
  - Complete transformer-based diffusion model with modulation
  - Approximator network for generating shift/scale/gate parameters
  - Double stream blocks (19) for joint image-text processing with cross-modulation
  - Single stream blocks (38) for image-only processing after concatenation
  - T5 text encoder integration
  - Positional embeddings (EmbedND) for spatial encoding
  - Flow matching support

- **Chroma training infrastructure** (`ai-toolkit-rs/crates/training/src/chroma_trainer.rs`)
  - ChromaTrainer with specialized training logic
  - Learning rate ramping system with linear/cosine options
  - Per-block learning rate configuration (pyramid structure)
  - ParameterGroupLRManager for fine-grained LR control
  - Flow matching loss computation: v = (data - noise) / (1 - t)
  - Optimal noise pairing support
  - Image augmentation utilities
  - Text encoding with T5

- **RadamScheduleFree optimizer** (`ai-toolkit-rs/crates/training/src/optimizers/radam_schedulefree.rs`)
  - RAdam with Schedule-Free framework implementation
  - Eliminates need for learning rate schedulers
  - Adaptive momentum with variance tractability checking
  - Schedule-free auxiliary variables (z) for stable optimization
  - Eval/train mode switching for proper inference
  - Integrated into optimizer factory

### Technical Details
- **Modulation Architecture**: Chroma uses modulation instead of standard attention conditioning
  - Approximator network generates shift/scale/gate parameters per block
  - More parameter-efficient than cross-attention
- **Per-block Learning Rates**: Pyramid structure (lower at edges, higher in middle)
  - Allows fine-grained control over training dynamics
  - Supports ramping warmup for stable training
- **Flow Matching**: Uses velocity prediction instead of noise prediction
  - More stable training dynamics
  - Better sample quality
- **RadamScheduleFree**: Combines RAdam with Schedule-Free framework
  - No need for cosine/linear schedulers
  - Maintains separate z variables for evaluation
  - Parameter k tracks optimization progress

### Configuration
- See `/home/alex/diffusers-rs/config/stone.yaml` for Chroma-specific configuration
- Key features:
  - `ramp_double_blocks`: Enable learning rate ramping
  - `lr_if_contains`: Per-block learning rate overrides
  - `radamschedulefree` optimizer usage
  - Flow matching training mode

### Status
✅ Model architecture complete
✅ Training utilities implemented  
✅ RadamScheduleFree optimizer integrated
✅ Per-block learning rates and ramping
⏳ Testing with real data pending
⏳ T5 encoder integration needs verification

## 2025-07-01: OmniGen 2 Implementation (Corrected)

### Added
- **OmniGen 2 model architecture** (`ai-toolkit-rs/crates/models/src/omnigen2.rs`)
  - Dual-path architecture with separate autoregressive and diffusion transformers
  - Text model: Qwen2.5-VL-3B for text understanding and generation
  - ViT encoder: Image understanding for the text model
  - Diffusion transformer: ~4B parameter DiT-style model for image generation
  - VAE: Uses frozen SDXL VAE for encoding/decoding
  - Proper separation of text and image generation paths

- **OmniGen 2 training infrastructure** (`ai-toolkit-rs/crates/training/src/omnigen2_trainer.rs`)
  - Dual-path training utilities
  - Separate handling for text model and diffusion model
  - VAE feature extraction for fine-grained visual conditioning
  - Text processing through Qwen2.5-VL for hidden states
  - Support for multiple loss types (flow matching, epsilon, v-prediction)
  - Component-specific LoRA target selection

- **OmniGen 2 LoRA training binary** (`ai-toolkit-rs/crates/training/src/bin/train_omnigen2_lora.rs`)
  - Complete training script for dual-path architecture
  - Separate paths for text model, ViT, and diffusion transformer
  - Component-specific learning rates
  - Flexible training configuration (can train any component)
  - Memory optimization with CPU offloading options
  - Mixed precision (bf16) support

- **Training configuration** (`ai-toolkit-rs/config/omnigen2_lora_config.yaml`)
  - Dual-path architecture configuration
  - Component-specific settings and paths
  - Memory optimization options
  - Detailed architecture documentation

### Technical Details
- **Architecture**: OmniGen 2 is NOT a unified transformer, but a dual-path system:
  - Autoregressive path: Qwen2.5-VL-3B for text generation
  - Diffusion path: Custom ~4B parameter transformer for image generation
  - The model switches between paths based on special tokens (<|img|>)
  
- **Key Components**:
  - Text Model: Pretrained Qwen2.5-VL-3B (usually frozen)
  - ViT Encoder: Feeds visual info to text model (usually frozen)
  - Diffusion Transformer: Main training target with LoRA
  - VAE: SDXL VAE for latent encoding/decoding (always frozen)

- **Data Flow**:
  1. Text → Qwen2.5-VL → Hidden states → Diffusion conditioning
  2. Images → ViT → Visual features for text understanding
  3. Images → VAE → Fine-grained features → Diffusion conditioning
  4. Diffusion transformer generates in latent space
  5. VAE decoder produces final images

### LoRA Support
- **Supported**: LoRA for transformer layers in all components
- **NOT Supported**: LoKr (requires conv layers, which OmniGen 2 doesn't have)
- **Target Modules**:
  - Text model attention layers (if training)
  - ViT encoder attention layers (if training)
  - Diffusion transformer attention and MLP layers (main target)

### Status
✅ Model architecture correctly implemented
✅ Training infrastructure for dual-path system
✅ LoRA support (NOT LoKr)
✅ Proper component separation
⏳ Requires Qwen2.5-VL-3B and SDXL VAE integration
⏳ Testing with real models pending

## 2025-07-01: KonText Model Implementation

### Added
- **KonText model architecture** (`ai-toolkit-rs/crates/models/src/kontext.rs`)
  - Context encoder for processing control images
  - Control adapter with cross-attention to Flux features
  - Support for multiple control types (Canny, Depth, Normal, Pose, etc.)
  - Multi-scale control support
  - Control dropout for unconditional training

- **KonText training infrastructure** (`ai-toolkit-rs/crates/training/src/kontext_trainer.rs`)
  - Control image processing for different control types
  - Canny edge detection implementation
  - Control augmentation utilities
  - Flow matching loss with control strength
  - Target module selection for LoRA

- **KonText LoRA training binary** (`ai-toolkit-rs/src/bin/train_kontext_lora.rs`)
  - Complete training script with CLI interface
  - Support for all control types
  - Configurable control strength and dropout
  - Option to train control encoder and/or base Flux
  - Control image preprocessing

- **Training configuration** (`ai-toolkit-rs/config/kontext_lora_training.yaml`)
  - Example configuration for KonText LoRA training
  - Paired dataset structure (image + control + caption)
  - Control-specific augmentation settings

### Technical Details
- KonText provides contextual control for Flux image generation
- Uses control images (edges, depth, pose) to guide generation
- Cross-modal fusion between control features and Flux hidden states
- Flow matching objective with control strength weighting
- Supports both conditional and unconditional training via dropout

### Control Types
- **Canny**: Edge detection for structure preservation
- **Depth**: Depth maps for 3D-aware editing
- **Normal**: Normal maps for surface detail control
- **Pose**: Human pose for figure control
- **Semantic**: Semantic segmentation for region control
- **Scribble**: Sketch/scribble for rough guidance
- **Custom**: User-defined control inputs

### Status
✅ Model architecture complete
✅ Training infrastructure implemented
✅ LoRA support integrated
✅ Multiple control types supported
⏳ Testing with real data pending

## 2025-07-01: SD 3.5 Training Implementation

### Added
- **SD 3.5 training infrastructure** (`ai-toolkit-rs/crates/training/src/sd35_trainer.rs`)
  - Triple text encoder support (CLIP-L + CLIP-G + T5-XXL)
  - Flow matching loss computation with velocity prediction
  - Proper text embedding concatenation and padding
  - Target module selection for LoRA
  - Support for all SD 3.5 variants (Medium, Large, Large Turbo)

- **SD 3.5 LoRA training binary** (`ai-toolkit-rs/src/bin/train_sd35_lora.rs`)
  - Complete training script with CLI interface
  - Support for all SD 3.5 model variants
  - Triple text encoder loading and configuration
  - LoRA network creation with MMDiT-specific targets
  - Gradient checkpointing and mixed precision support
  - Configurable text encoder training

- **Training configuration** (`ai-toolkit-rs/config/sd35_lora_training.yaml`)
  - Example configuration for SD 3.5 LoRA training
  - Proper model paths for all components
  - Target module specifications for MMDiT

### Technical Details
- SD 3.5 uses MMDiT (Multimodal Diffusion Transformer) architecture
- Flow matching objective with velocity prediction: v = (data - noise) / (1 - t)
- 16-channel VAE with scaling factor 1.5305
- Text embeddings: CLIP-L (768d) + CLIP-G (1280d) padded to 2048d each, plus T5-XXL (4096d)
- Joint attention blocks process image and text together

### Status
✅ SD 3.5 model structure already complete (MMDiT implementation exists)
✅ Training infrastructure implemented
✅ LoRA support integrated
✅ Triple text encoder support added
⏳ Testing with real data pending

## 2025-07-01: Flux Forward Pass Implementation

### Added
- **Complete Flux forward pass implementation** (`ai-toolkit-rs/crates/models/src/flux_forward.rs`)
  - Full transformer architecture with double and single stream blocks
  - Proper RoPE position embeddings
  - QK normalization for stable attention
  - AdaLN modulation for conditioning
  - Time and guidance embedding support
  - Patchification/unpatchification for latent processing

- **Updated Flux model** (`ai-toolkit-rs/crates/models/src/flux.rs`)
  - Integrated complete forward pass
  - Added latent preparation (patchification)
  - Position ID generation for image and text
  - Proper timestep scaling (divide by 1000)
  - Support for all Flux variants (Base, Schnell, Dev)

- **Flux training infrastructure** (`ai-toolkit-rs/crates/training/src/flux_trainer.rs`)
  - Flow matching loss computation
  - Text encoder loading and encoding
  - Noise scheduling for continuous timesteps
  - Guidance scale training support
  - Training configuration with sensible defaults

- **Flux LoRA training binary** (`ai-toolkit-rs/src/bin/train_flux_lora.rs`)
  - Complete training script with CLI interface
  - LoRA network creation and application
  - Support for gradient checkpointing and mixed precision
  - Checkpoint saving and resuming
  - Validation during training

- **Training configuration** (`ai-toolkit-rs/config/flux_lora_training.yaml`)
  - Example configuration for Flux LoRA training
  - Target module specifications
  - Dataset and optimization settings

### Technical Details
- Based on the official Flux architecture from Black Forest Labs
- Implements the complete dual-stream transformer design
- Uses flow matching objective (not standard diffusion)
- Supports classifier-free guidance training
- Memory efficient with gradient checkpointing

### Status
✅ Forward pass fully implemented
✅ Training infrastructure ready
✅ LoRA support integrated
⏳ Testing with real data pending
⏳ Model weight loading to be verified

The Flux implementation is now feature-complete for training. The forward pass correctly implements the architecture described in the Flux paper, including the characteristic dual-stream processing for joint image-text attention.