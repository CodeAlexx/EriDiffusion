use super::keymap::Sd35KeyMap;
use crate::chroma::weights::MmapWeightProvider as GenericProvider;

pub type Sd35WeightProvider = GenericProvider<Sd35KeyMap>;
