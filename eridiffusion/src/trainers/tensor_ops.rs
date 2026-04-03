use flame_core::tensor_ops_extended;
use flame_core::{DType, Error, Result, Shape, Tensor};
use num_traits;

// Common tensor extension methods
pub trait TensorOpsExt {
    fn to_scalar<T: num_traits::NumCast>(&self) -> Result<T>;
    fn clamp(&self, min: f64, max: f64) -> Result<Tensor>;
    fn dim(&self, axis: i32) -> Result<usize>;
    fn flatten_to(&self, dim: usize) -> Result<Tensor>;
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor>;
    fn get_on_dim(&self, dim: i64, index: i64) -> Result<Tensor>;
    fn t(&self) -> Result<Tensor>;
    fn to_vec1<T: num_traits::NumCast>(&self) -> Result<Vec<T>>;
}

impl TensorOpsExt for Tensor {
    fn to_scalar<T: num_traits::NumCast>(&self) -> Result<T> {
        // Get the first element and convert to requested type
        let data = self.to_vec1::<f32>()?;
        if data.is_empty() {
            return Err(flame_core::Error::InvalidOperation(
                "Cannot convert empty tensor to scalar".to_string(),
            ));
        }
        num_traits::cast::cast(data[0]).ok_or_else(|| {
            flame_core::Error::InvalidOperation("Failed to cast scalar value".into())
        })
    }

    fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        let min_tensor = Tensor::full(self.shape().clone(), min as f32, self.device().clone())?;
        let max_tensor = Tensor::full(self.shape().clone(), max as f32, self.device().clone())?;
        self.maximum(&min_tensor)?.minimum(&max_tensor)
    }

    fn dim(&self, axis: i32) -> Result<usize> {
        let shape = self.shape();
        let ndim = shape.rank() as i32;
        let axis = if axis < 0 { ndim + axis } else { axis };
        if axis < 0 || axis >= ndim {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Axis {} out of range for tensor with {} dimensions",
                axis, ndim
            )));
        }
        Ok(shape.dims()[axis as usize])
    }

    fn flatten_to(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();
        let dims = shape.dims();
        if dim >= dims.len() {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        let mut new_shape = vec![];
        let mut flattened_size = 1;
        for i in 0..=dim {
            new_shape.push(dims[i]);
        }
        for i in (dim + 1)..dims.len() {
            flattened_size *= dims[i];
        }
        if flattened_size > 1 {
            new_shape.push(flattened_size);
        }

        self.reshape(&new_shape)
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
        // FLAME's narrow method
        Tensor::narrow(self, dim, start, length)
    }

    fn get_on_dim(&self, dim: i64, index: i64) -> Result<Tensor> {
        // Use narrow to get a single element along the dimension, then squeeze
        self.narrow(dim as usize, index as usize, 1)?.squeeze_dim(dim as usize)
    }

    fn t(&self) -> Result<Tensor> {
        let rank = self.shape().rank();
        if rank < 2 {
            return Err(flame_core::Error::InvalidOperation(
                "Cannot transpose tensor with rank < 2".to_string(),
            ));
        }
        self.transpose_dims(rank - 2, rank - 1)
    }

    fn to_vec1<T: num_traits::NumCast>(&self) -> Result<Vec<T>> {
        // First convert to F32 if needed (FLAME's to_vec returns Vec<f32>)
        let f32_tensor = match self.dtype() {
            flame_core::DType::F32 => self.clone(),
            flame_core::DType::F16 | flame_core::DType::BF16 => {
                self.to_dtype(flame_core::DType::F32)?
            }
            _ => {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "Unsupported dtype for to_vec1: {:?}",
                    self.dtype()
                )));
            }
        };

        // Use FLAME's built-in to_vec method
        let vec_f32 = f32_tensor.to_vec()?;

        // Convert to target type
        vec_f32
            .into_iter()
            .map(|v| {
                T::from(v).ok_or_else(|| {
                    flame_core::Error::InvalidOperation("Failed to cast value".into())
                })
            })
            .collect()
    }
}
