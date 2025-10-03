
#![allow(dead_code)]
#[derive(Clone, Debug)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
    pub dtype: Option<&'static str>,
}
impl HostTensor {
    pub fn zeros(shape: &[usize], bytes_per_el: usize, dtype: Option<&'static str>) -> Self {
        let n: usize = shape.iter().product();
        Self { shape: shape.to_vec(), bytes: vec![0u8; n * bytes_per_el], dtype }
    }
}
