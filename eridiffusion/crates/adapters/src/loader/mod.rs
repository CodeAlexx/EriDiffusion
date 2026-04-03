pub mod safetensors_lyco;

use hashbrown::HashMap;

#[derive(Default)]
pub struct BaseShapes { pub by_target: HashMap<String, (Vec<i64>, bool)> }

