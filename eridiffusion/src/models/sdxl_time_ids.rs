use anyhow::Context;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;

pub struct TimeIdsGenerator;

impl TimeIdsGenerator {
    /// Generate time IDs for a single image
    pub fn generate_single(
        config: &TimeIdsConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        config.validate()?;
        config.to_tensor_dtype(device, dtype)
    }

    /// Generate time IDs for batch processing
    pub fn generate_batch(
        configs: &[TimeIdsConfig],
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        if configs.is_empty() {
            return Err(Error::InvalidOperation(
                "Cannot generate batch time_ids from empty configs".to_string(),
            ));
        }

        let batch_size = configs.len();
        let mut time_ids_vec = Vec::with_capacity(batch_size * 6);

        for config in configs {
            config.validate()?;
            time_ids_vec.extend_from_slice(&[
                config.original_height,
                config.original_width,
                config.crop_top,
                config.crop_left,
                config.target_height,
                config.target_width,
            ]);
        }

        let tensor = Tensor::from_vec(
            time_ids_vec,
            Shape::from_dims(&[batch_size, 6]),
            device.cuda_device().clone(),
        )?;
        tensor.to_dtype(dtype).map_err(|e| {
            Error::InvalidOperation(format!("Failed to convert batch time_ids dtype: {}", e))
        })
    }

    /// Generate time IDs for classifier-free guidance
    /// Returns concatenated time_ids for negative and positive prompts
    pub fn generate_for_cfg(
        config: &TimeIdsConfig,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        config.validate()?;

        let single_time_ids = config.to_tensor_dtype(device, dtype)?;

        // Expand to batch size
        let time_ids = if batch_size > 1 {
            single_time_ids.unsqueeze(0)?.broadcast_to(&Shape::from_dims(&[batch_size, 6]))?
        } else {
            single_time_ids.unsqueeze(0)?
        };

        // Duplicate for negative and positive prompts
        Tensor::cat(&[&time_ids, &time_ids], 0)
    }

    /// Generate time IDs with aesthetic scores (for models fine-tuned with aesthetic conditioning)
    pub fn generate_with_aesthetic_score(
        config: &TimeIdsConfig,
        aesthetic_score: f32,
        negative_aesthetic_score: f32,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        config.validate()?;

        // Validate aesthetic scores
        if aesthetic_score < 0.0 || aesthetic_score > 10.0 {
            return Err(Error::InvalidOperation(
                "Aesthetic score must be between 0 and 10".to_string(),
            ));
        }
        if negative_aesthetic_score < 0.0 || negative_aesthetic_score > 10.0 {
            return Err(Error::InvalidOperation(
                "Negative aesthetic score must be between 0 and 10".to_string(),
            ));
        }

        let positive_ids = vec![
            config.original_height,
            config.original_width,
            config.crop_top,
            config.crop_left,
            config.target_height,
            config.target_width,
            aesthetic_score,
        ];

        let negative_ids = vec![
            config.original_height,
            config.original_width,
            config.crop_top,
            config.crop_left,
            config.target_height,
            config.target_width,
            negative_aesthetic_score,
        ];

        let pos_tensor = Tensor::from_vec(
            positive_ids,
            Shape::from_dims(&[1, 7]),
            device.cuda_device().clone(),
        )?
        .to_dtype(dtype)?;
        let neg_tensor = Tensor::from_vec(
            negative_ids,
            Shape::from_dims(&[1, 7]),
            device.cuda_device().clone(),
        )?
        .to_dtype(dtype)?;

        // Expand to batch size
        let pos_batch = if batch_size > 1 {
            pos_tensor.broadcast_to(&Shape::from_dims(&[batch_size, 7]))?
        } else {
            pos_tensor
        };

        let neg_batch = if batch_size > 1 {
            neg_tensor.broadcast_to(&Shape::from_dims(&[batch_size, 7]))?
        } else {
            neg_tensor
        };

        // Concatenate negative first, then positive (for CFG)
        Tensor::cat(&[&neg_batch, &pos_batch], 0)
    }

    /// Generate time IDs for multiple images with different resolutions
    pub fn generate_multi_resolution(
        configs: &[TimeIdsConfig],
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<Tensor>> {
        configs.iter().map(|config| Self::generate_single(config, device, dtype)).collect()
    }

    /// Generate time IDs from actual crop parameters (useful for img2img)
    pub fn from_image_crop(
        original_height: usize,
        original_width: usize,
        crop_top: usize,
        crop_left: usize,
        crop_height: usize,
        crop_width: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let config = TimeIdsConfig::with_custom_crop(
            original_height,
            original_width,
            crop_top as f32,
            crop_left as f32,
            crop_height,
            crop_width,
        );

        Self::generate_single(&config, device, dtype)
    }
}

/// Helper for common SDXL resolutions and aspect ratios
pub struct SDXLResolutions;

impl SDXLResolutions {
    /// Standard SDXL square resolution
    pub const STANDARD_SIZE: usize = 1024;

    /// Get time IDs for standard SDXL resolution (1024x1024)
    pub fn standard(device: &Device, dtype: DType) -> Result<Tensor> {
        let config = TimeIdsConfig::new(Self::STANDARD_SIZE, Self::STANDARD_SIZE);
        TimeIdsGenerator::generate_single(&config, device, dtype)
    }

    /// Get time IDs for common aspect ratios at standard resolution
    pub fn for_aspect_ratio(aspect_ratio: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let (height, width) = match aspect_ratio {
            "1:1" => (1024, 1024),
            "16:9" => (576, 1024),
            "9:16" => (1024, 576),
            "4:3" => (896, 1152),
            "3:4" => (1152, 896),
            "21:9" => (448, 1024),
            "9:21" => (1024, 448),
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "Unsupported aspect ratio: {}",
                    aspect_ratio
                )))
            }
        };

        let config = TimeIdsConfig::new(height, width);
        TimeIdsGenerator::generate_single(&config, device, dtype)
    }

    /// Get time IDs for a specific resolution
    pub fn for_resolution(
        height: usize,
        width: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        // Validate resolution is appropriate for SDXL
        if height < 512 || width < 512 {
            return Err(Error::InvalidOperation(
                "SDXL requires minimum dimension of 512px".to_string(),
            ));
        }
        if height > 2048 || width > 2048 {
            return Err(Error::InvalidOperation("SDXL maximum dimension is 2048px".to_string()));
        }
        if (height * width) > (2048 * 2048) {
            return Err(Error::InvalidOperation(
                "Total pixel count exceeds SDXL maximum".to_string(),
            ));
        }

        let config = TimeIdsConfig::new(height, width);
        TimeIdsGenerator::generate_single(&config, device, dtype)
    }

    /// Get all standard SDXL training resolutions
    pub fn all_training_resolutions() -> Vec<(usize, usize)> {
        vec![
            (1024, 1024), // 1:1
            (1152, 896),  // 9:7
            (896, 1152),  // 7:9
            (1216, 832),  // 19:13
            (832, 1216),  // 13:19
            (1344, 768),  // 7:4
            (768, 1344),  // 4:7
            (1536, 640),  // 12:5
            (640, 1536),  // 5:12
        ]
    }

    /// Calculate the nearest SDXL-compatible resolution for given dimensions
    pub fn nearest_compatible_resolution(
        target_height: usize,
        target_width: usize,
    ) -> (usize, usize) {
        let target_pixels = target_height * target_width;
        let target_ratio = target_width as f32 / target_height as f32;

        let mut best_resolution = (1024, 1024);
        let mut best_score = f32::MAX;

        for &(h, w) in &Self::all_training_resolutions() {
            let pixels = h * w;
            let ratio = w as f32 / h as f32;

            // Score based on pixel count difference and aspect ratio difference
            let pixel_diff = ((pixels as f32 - target_pixels as f32).abs()) / target_pixels as f32;
            let ratio_diff = (ratio - target_ratio).abs();
            let score = pixel_diff + ratio_diff * 2.0; // Weight aspect ratio more heavily

            if score < best_score {
                best_score = score;
                best_resolution = (h, w);
            }
        }

        best_resolution
    }
}

/// Helper for integrating time IDs with SDXL pipeline
pub struct SDXLConditioningHelper;

impl SDXLConditioningHelper {
    /// Prepare added conditioning kwargs for SDXL UNet
    pub fn prepare_added_cond_kwargs(
        text_embeds: &Tensor,
        time_ids: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut added_cond_kwargs = HashMap::new();

        // Validate tensor shapes
        let text_embeds_shape = text_embeds.shape();
        if text_embeds_shape.dims().len() != 2 || text_embeds_shape.dims()[1] != 1280 {
            return Err(Error::InvalidOperation(format!(
                "Invalid text_embeds shape: expected (batch_size, 1280), got {:?}",
                text_embeds_shape
            )));
        }

        let time_ids_shape = time_ids.shape();
        if time_ids_shape.dims().len() != 2 || time_ids_shape.dims()[1] != 6 {
            return Err(Error::InvalidOperation(format!(
                "Invalid time_ids shape: expected (batch_size, 6), got {:?}",
                time_ids_shape
            )));
        }

        added_cond_kwargs.insert("text_embeds".to_string(), text_embeds.clone());
        added_cond_kwargs.insert("time_ids".to_string(), time_ids.clone());

        Ok(added_cond_kwargs)
    }

    /// Prepare conditioning for inference with classifier-free guidance
    pub fn prepare_cfg_conditioning(
        pooled_embeds_pos: &Tensor, // Positive prompt pooled embeddings
        pooled_embeds_neg: &Tensor, // Negative prompt pooled embeddings
        height: usize,
        width: usize,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        // Validate input shapes
        let pos_shape = pooled_embeds_pos.shape();
        let neg_shape = pooled_embeds_neg.shape();

        if pos_shape != neg_shape {
            return Err(Error::InvalidOperation(
                "Positive and negative embeddings must have same shape".to_string(),
            ));
        }

        if pos_shape.dims().len() != 2 || pos_shape.dims()[1] != 1280 {
            return Err(Error::InvalidOperation(format!(
                "Invalid pooled embeddings shape: expected (batch_size, 1280), got {:?}",
                pos_shape
            )));
        }

        // Concatenate text embeds for CFG (negative first, then positive)
        let text_embeds_cfg = Tensor::cat(&[pooled_embeds_neg, pooled_embeds_pos], 0)?;

        // Generate time IDs for CFG
        let time_config = TimeIdsConfig::new(height, width);
        let time_ids_cfg =
            TimeIdsGenerator::generate_for_cfg(&time_config, batch_size, device, dtype)?;

        let mut added_cond_kwargs = HashMap::new();
        added_cond_kwargs.insert("text_embeds".to_string(), text_embeds_cfg);
        added_cond_kwargs.insert("time_ids".to_string(), time_ids_cfg);

        Ok(added_cond_kwargs)
    }

    /// Prepare conditioning with custom crop parameters
    pub fn prepare_cfg_conditioning_with_crop(
        pooled_embeds_pos: &Tensor,
        pooled_embeds_neg: &Tensor,
        time_config: &TimeIdsConfig,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        // Concatenate text embeds
        let text_embeds_cfg = Tensor::cat(&[pooled_embeds_neg, pooled_embeds_pos], 0)?;

        // Generate time IDs with custom config
        let time_ids_cfg =
            TimeIdsGenerator::generate_for_cfg(time_config, batch_size, device, dtype)?;

        let mut added_cond_kwargs = HashMap::new();
        added_cond_kwargs.insert("text_embeds".to_string(), text_embeds_cfg);
        added_cond_kwargs.insert("time_ids".to_string(), time_ids_cfg);

        Ok(added_cond_kwargs)
    }

    /// Extract time IDs from added_cond_kwargs (useful for debugging/logging)
    pub fn extract_time_ids(
        added_cond_kwargs: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Vec<TimeIdsConfig>> {
        let time_ids = added_cond_kwargs.get("time_ids").ok_or_else(|| {
            Error::InvalidOperation("time_ids not found in added_cond_kwargs".into())
        })?;

        let shape = time_ids.shape();
        if shape.dims().len() != 2 || shape.dims()[1] != 6 {
            return Err(Error::InvalidOperation(format!(
                "Invalid time_ids shape: expected (batch_size, 6), got {:?}",
                shape
            )));
        }

        let batch_size = shape.dims()[0];
        let time_ids_vec = time_ids.to_vec2::<f32>()?;

        let mut configs = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            configs.push(TimeIdsConfig {
                original_height: time_ids_vec[i][0],
                original_width: time_ids_vec[i][1],
                crop_top: time_ids_vec[i][2],
                crop_left: time_ids_vec[i][3],
                target_height: time_ids_vec[i][4],
                target_width: time_ids_vec[i][5],
            });
        }

        Ok(configs)
    }
}

// FLAME uses flame_core::device::Device instead of Device

/// Time IDs configuration for SDXL
/// Contains 6 values: original_height, original_width, crop_top, crop_left, target_height, target_width
#[derive(Debug, Clone, Copy)]
pub struct TimeIdsConfig {
    /// Original image height before any processing
    pub original_height: f32,
    /// Original image width before any processing
    pub original_width: f32,
    /// Top coordinate of the crop box
    pub crop_top: f32,
    /// Left coordinate of the crop box
    pub crop_left: f32,
    /// Target height for generation
    pub target_height: f32,
    /// Target width for generation
    pub target_width: f32,
}

impl TimeIdsConfig {
    /// Create a new TimeIdsConfig with default values matching target size (no cropping)
    pub fn new(target_height: usize, target_width: usize) -> Self {
        Self {
            original_height: target_height as f32,
            original_width: target_width as f32,
            crop_top: 0.0,
            crop_left: 0.0,
            target_height: target_height as f32,
            target_width: target_width as f32,
        }
    }

    /// Create a TimeIdsConfig for center-cropped images
    pub fn with_center_crop(
        original_height: usize,
        original_width: usize,
        target_height: usize,
        target_width: usize,
    ) -> Self {
        let crop_top = ((original_height.saturating_sub(target_height)) / 2) as f32;
        let crop_left = ((original_width.saturating_sub(target_width)) / 2) as f32;

        Self {
            original_height: original_height as f32,
            original_width: original_width as f32,
            crop_top,
            crop_left,
            target_height: target_height as f32,
            target_width: target_width as f32,
        }
    }

    /// Create a TimeIdsConfig with custom crop coordinates
    pub fn with_custom_crop(
        original_height: usize,
        original_width: usize,
        crop_top: f32,
        crop_left: f32,
        target_height: usize,
        target_width: usize,
    ) -> Self {
        Self {
            original_height: original_height as f32,
            original_width: original_width as f32,
            crop_top,
            crop_left,
            target_height: target_height as f32,
            target_width: target_width as f32,
        }
    }

    /// Convert to tensor representation
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let time_ids = vec![
            self.original_height,
            self.original_width,
            self.crop_top,
            self.crop_left,
            self.target_height,
            self.target_width,
        ];

        Tensor::from_vec(time_ids, Shape::from_dims(&[6]), device.cuda_device().clone())
    }

    /// Convert to tensor with specific dtype
    pub fn to_tensor_dtype(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let tensor = self.to_tensor(device)?;
        tensor.to_dtype(dtype).map_err(|e| {
            Error::InvalidOperation(format!("Failed to convert time_ids dtype: {}", e))
        })
    }

    /// Validate that crop and target dimensions are within original dimensions
    pub fn validate(&self) -> Result<()> {
        if self.crop_top < 0.0 || self.crop_left < 0.0 {
            return Err(Error::InvalidOperation(
                "Crop coordinates must be non-negative".to_string(),
            ));
        }

        if self.crop_top + self.target_height > self.original_height {
            return Err(Error::InvalidOperation(format!(
                "Crop top ({}) + target height ({}) exceeds original height ({})",
                self.crop_top, self.target_height, self.original_height
            )));
        }

        if self.crop_left + self.target_width > self.original_width {
            return Err(Error::InvalidOperation(format!(
                "Crop left ({}) + target width ({}) exceeds original width ({})",
                self.crop_left, self.target_width, self.original_width
            )));
        }

        Ok(())
    }
}
