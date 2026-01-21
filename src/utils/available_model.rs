use std::str::FromStr;
use strum::{Display, EnumIter, EnumProperty, EnumString, IntoEnumIterator, IntoStaticStr};

#[derive(Debug, PartialEq, Eq, EnumProperty, EnumString, EnumIter, Display, IntoStaticStr)]
#[strum(serialize_all = "kebab-case")]
pub enum AvailableModel {
  #[strum(props(hf = "onnx-community/chatterbox-multilingual-ONNX"))]
  ChatterboxMultilingual,
  #[strum(props(hf = "onnx-community/Kokoro-82M-v1.0-ONNX"))]
  Kokoro,
}

impl AvailableModel {
  #[must_use]
  pub fn model_name(&self) -> &str {
    self.into()
  }

  #[must_use]
  pub fn hf_id(&self) -> &str {
    self.get_str("hf").unwrap_or_default()
  }

  #[must_use]
  pub fn from_model_name(model_id: &str) -> Option<Self> {
    Self::from_str(model_id).ok()
  }

  #[must_use]
  pub fn from_hf_id(hf_id: &str) -> Option<Self> {
    Self::iter().find(|variant| variant.hf_id() == hf_id)
  }
}
