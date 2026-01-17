pub enum AvailableModel {
  ChatterboxMultilingual,
  Kokoro,
}

impl AvailableModel {
  pub fn model_name(&self) -> &str {
    match self {
      Self::ChatterboxMultilingual => "chatterbox-multilingual",
      Self::Kokoro => "kokoro",
    }
  }

  pub fn hf_id(&self) -> &str {
    match self {
      Self::ChatterboxMultilingual => "onnx-community/chatterbox-multilingual-ONNX",
      Self::Kokoro => "onnx-community/Kokoro-82M-v1.0-ONNX",
    }
  }

  pub fn from_model_name(model_id: &str) -> Option<Self> {
    match model_id {
      "chatterbox-multilingual" => Some(Self::ChatterboxMultilingual),
      "kokoro" => Some(Self::Kokoro),
      _ => None,
    }
  }

  pub fn from_hf_id(hf_id: &str) -> Option<Self> {
    match hf_id {
      "onnx-community/chatterbox-multilingual-ONNX" => Some(Self::ChatterboxMultilingual),
      "onnx-community/Kokoro-82M-v1.0-ONNX" => Some(Self::Kokoro),
      _ => None,
    }
  }
}
