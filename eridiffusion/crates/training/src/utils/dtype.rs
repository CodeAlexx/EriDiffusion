use eridiffusion_core::{Error, Result};
use flame_core::{DType, Tensor};

/// Cast tensor to the requested dtype while preserving the computation graph.
pub fn cast_preserve_grad(x: &Tensor, dtype: DType) -> Result<Tensor> {
    if x.dtype() == dtype {
        return Ok(x.clone());
    }
    let scaled = x.mul_scalar(1.0f32).map_err(Error::from)?;
    scaled.to_dtype(dtype).map_err(Error::from)
}
