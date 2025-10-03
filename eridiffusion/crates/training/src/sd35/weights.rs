use crate::chroma::weights::MmapWeightProvider as GenericProvider;
use super::keymap::Sd35KeyMap;

pub type Sd35WeightProvider = GenericProvider<Sd35KeyMap>;

