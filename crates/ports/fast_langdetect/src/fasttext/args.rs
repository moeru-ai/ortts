//! Port of fastText `src/args.{h,cc}` (the parts that `loadModel` reads).

use std::io::Read;

use super::read::{read_f64, read_i32};

/// Port of fastText `enum class model_name` (`args.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelName {
  /// `model_name::cbow` (1).
  Cbow,
  /// `model_name::sg` (2).
  Sg,
  /// `model_name::sup` (3).
  Sup,
  /// Any other stored value.
  Unknown(i32),
}

impl ModelName {
  fn from_i32(v: i32) -> Self {
    match v {
      1 => Self::Cbow,
      2 => Self::Sg,
      3 => Self::Sup,
      other => Self::Unknown(other),
    }
  }
}

/// Port of fastText `enum class loss_name` (`args.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossName {
  /// `loss_name::hs` (1).
  Hs,
  /// `loss_name::ns` (2).
  Ns,
  /// `loss_name::softmax` (3).
  Softmax,
  /// `loss_name::ova` (4).
  Ova,
  /// Any other stored value.
  Unknown(i32),
}

impl LossName {
  fn from_i32(v: i32) -> Self {
    match v {
      1 => Self::Hs,
      2 => Self::Ns,
      3 => Self::Softmax,
      4 => Self::Ova,
      other => Self::Unknown(other),
    }
  }
}

/// Port of fastText `class Args` (only the fields stored in a model file, plus
/// `label` and `qout`).
#[derive(Debug, Clone)]
pub struct Args {
  /// `Args::dim`.
  pub dim: i32,
  /// `Args::ws`.
  pub ws: i32,
  /// `Args::epoch`.
  pub epoch: i32,
  /// `Args::minCount`.
  pub min_count: i32,
  /// `Args::neg`.
  pub neg: i32,
  /// `Args::wordNgrams`.
  pub word_ngrams: i32,
  /// `Args::loss`.
  pub loss: LossName,
  /// `Args::model`.
  pub model: ModelName,
  /// `Args::bucket`.
  pub bucket: i32,
  /// `Args::minn`.
  pub minn: i32,
  /// `Args::maxn`.
  pub maxn: i32,
  /// `Args::lrUpdateRate`.
  pub lr_update_rate: i32,
  /// `Args::t`.
  pub t: f64,
  /// `Args::label` (not stored in the file; default `"__label__"`).
  pub label: String,
  /// `Args::qout`.
  pub qout: bool,
}

impl Args {
  /// Port of `Args::load(std::istream&)`.
  pub fn load<R: Read>(r: &mut R) -> std::io::Result<Self> {
    Ok(Self {
      dim: read_i32(r)?,
      ws: read_i32(r)?,
      epoch: read_i32(r)?,
      min_count: read_i32(r)?,
      neg: read_i32(r)?,
      word_ngrams: read_i32(r)?,
      loss: LossName::from_i32(read_i32(r)?),
      model: ModelName::from_i32(read_i32(r)?),
      bucket: read_i32(r)?,
      minn: read_i32(r)?,
      maxn: read_i32(r)?,
      lr_update_rate: read_i32(r)?,
      t: read_f64(r)?,
      label: "__label__".to_string(),
      qout: false,
    })
  }
}
