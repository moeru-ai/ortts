use std::str::FromStr;
use strum::{Display, EnumIter, EnumProperty, EnumString, IntoEnumIterator, IntoStaticStr};

#[derive(Debug, PartialEq, EnumProperty, EnumString, EnumIter, Display, IntoStaticStr)]
#[strum(serialize_all = "kebab-case")]
pub enum AvailableModel {
  #[strum(props(hf = "onnx-community/chatterbox-multilingual-ONNX"))]
  ChatterboxMultilingual,
  #[strum(props(hf = "onnx-community/Kokoro-82M-v1.0-ONNX"))]
  Kokoro,
}

impl AvailableModel {
  pub fn model_name(&self) -> &str {
    self.into()
  }

  pub fn hf_id(&self) -> &str {
    self.get_str("hf").unwrap_or_default()
  }

  pub fn from_model_name(model_id: &str) -> Option<Self> {
    Self::from_str(model_id).ok()
  }

  pub fn from_hf_id(hf_id: &str) -> Option<Self> {
    Self::iter().find(|variant| variant.hf_id() == hf_id)
  }
}
