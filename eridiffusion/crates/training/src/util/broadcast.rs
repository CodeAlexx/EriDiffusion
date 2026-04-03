use eridiffusion_core::{Error, Result};
use flame_core::{Shape, Tensor};

pub fn broadcast_to_like(x: &Tensor, like: &Tensor) -> Result<Tensor> {
    x.broadcast_to(&Shape::from_dims(like.dims())).map_err(Error::from)
}
