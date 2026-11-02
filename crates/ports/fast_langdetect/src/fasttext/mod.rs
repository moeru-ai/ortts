//! Pure Rust port of the prediction path of the `fasttext` Python module from
//! fasttext-predict 0.9.2.4 (`fasttext/FastText.py` and the fastText 0.9.2 C++
//! sources it binds: `fasttext.cc`, `model.cc`, `dictionary.cc`, `loss.cc`,
//! `densematrix.cc`, `args.cc`).
//!
//! Only non-quantized supervised models with the hierarchical softmax loss
//! (such as `lid.176.bin`) are supported.

pub mod args;
pub mod densematrix;
pub mod dictionary;
mod heap;
pub mod loss;
mod read;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use args::{Args, LossName, ModelName};
use densematrix::{DenseMatrix, vector_mul, vector_zero};
use dictionary::Dictionary;
use loss::HierarchicalSoftmaxLoss;
use read::{read_bool, read_i32};

use crate::FastLangdetectError;

/// Port of `FASTTEXT_FILEFORMAT_MAGIC_INT32`.
pub const FASTTEXT_FILEFORMAT_MAGIC_INT32: i32 = 793712314;
/// Port of `FASTTEXT_VERSION`.
pub const FASTTEXT_VERSION: i32 = 12;
/// Port of `Model::kUnlimitedPredictions`.
pub const K_UNLIMITED_PREDICTIONS: i32 = -1;

/// Port of the C++ `fasttext::FastText` model wrapped by the Python
/// `fasttext.FastText._FastText` class.
#[derive(Debug, Clone)]
pub struct FastText {
  args: Args,
  dict: Dictionary,
  input: DenseMatrix,
  output: DenseMatrix,
  loss: HierarchicalSoftmaxLoss,
}

/// Port of `fasttext.load_model(path)` (`FastText::loadModel(const std::string&)`).
pub fn load_model(path: impl AsRef<Path>) -> Result<FastText, FastLangdetectError> {
  let path = path.as_ref();
  let file = File::open(path).map_err(|_| {
    FastLangdetectError::InvalidArgument(format!(
      "{} cannot be opened for loading!",
      path.display()
    ))
  })?;
  let mut r = BufReader::with_capacity(1 << 20, file);
  FastText::load(&mut r).map_err(|e| match e {
    FastLangdetectError::InvalidArgument(msg) if msg.is_empty() => {
      FastLangdetectError::InvalidArgument(format!("{} has wrong file format!", path.display()))
    }
    other => other,
  })
}

impl FastText {
  /// Port of `FastText::checkModel` + `FastText::loadModel(std::istream&)`.
  pub fn load<R: Read>(r: &mut R) -> Result<Self, FastLangdetectError> {
    let io = |e: std::io::Error| FastLangdetectError::ModelLoad(e.to_string());
    // checkModel(); an empty message is replaced by the path-specific one.
    let magic = read_i32(r).map_err(|_| FastLangdetectError::InvalidArgument(String::new()))?;
    if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32 {
      return Err(FastLangdetectError::InvalidArgument(String::new()));
    }
    let version = read_i32(r).map_err(|_| FastLangdetectError::InvalidArgument(String::new()))?;
    if version > FASTTEXT_VERSION {
      return Err(FastLangdetectError::InvalidArgument(String::new()));
    }

    let mut args = Args::load(r).map_err(io)?;
    if version == 11 && args.model == ModelName::Sup {
      // backward compatibility: old supervised models do not use char ngrams.
      args.maxn = 0;
    }
    let dict = Dictionary::load(args.clone(), r).map_err(io)?;
    let quant_input = read_bool(r).map_err(io)?;
    if quant_input {
      return Err(FastLangdetectError::ModelLoad(
        "quantized models (.ftz) are not supported by this port".into(),
      ));
    }
    let input = DenseMatrix::load(r).map_err(io)?;
    if dict.is_pruned() {
      return Err(FastLangdetectError::InvalidArgument(
                "Invalid model file.\nPlease download the updated model from www.fasttext.cc.\nSee issue #332 on Github for more information.\n".into(),
            ));
    }
    args.qout = read_bool(r).map_err(io)?;
    let output = DenseMatrix::load(r).map_err(io)?;

    // buildModel() / FastText::createLoss
    if args.model != ModelName::Sup {
      return Err(FastLangdetectError::ModelLoad(
        "only supervised models are supported by this port".into(),
      ));
    }
    let loss = match args.loss {
      LossName::Hs => HierarchicalSoftmaxLoss::new(&dict.get_counts(dictionary::EntryType::Label)),
      other => {
        return Err(FastLangdetectError::ModelLoad(format!(
          "loss {other:?} is not supported by this port (only hs)"
        )));
      }
    };
    Ok(Self {
      args,
      dict,
      input,
      output,
      loss,
    })
  }

  /// The model arguments (`FastText::getArgs`).
  pub fn get_args(&self) -> &Args {
    &self.args
  }

  /// The model dictionary (`FastText::getDictionary`).
  pub fn get_dictionary(&self) -> &Dictionary {
    &self.dict
  }

  /// Port of `Model::computeHidden`.
  fn compute_hidden(&self, input: &[i32], hidden: &mut [f32]) {
    vector_zero(hidden);
    for &it in input {
      self.input.add_row_to_vector(hidden, it);
    }
    // `hidden.mul(1.0 / input.size())`: double division narrowed to `real`.
    vector_mul(hidden, (1.0f64 / input.len() as f64) as f32);
  }

  /// Port of `FastText::predict(int32_t k, const std::vector<int32_t>& words,
  /// Predictions&, real threshold)` and `Model::predict`.
  fn predict_words(
    &self,
    mut k: i32,
    words: &[i32],
    threshold: f32,
  ) -> Result<Vec<(f32, i32)>, FastLangdetectError> {
    let mut heap = Vec::new();
    if words.is_empty() {
      return Ok(heap);
    }
    if k == K_UNLIMITED_PREDICTIONS {
      k = self.output.size(0) as i32;
    } else if k <= 0 {
      return Err(FastLangdetectError::InvalidArgument(
        "k needs to be 1 or higher!".into(),
      ));
    }
    heap.reserve(k as usize + 1);
    let mut hidden = vec![0f32; self.args.dim as usize];
    self.compute_hidden(words, &mut hidden);
    self
      .loss
      .predict(k, threshold, &mut heap, &hidden, &self.output);
    Ok(heap)
  }

  /// Port of `FastText::predictLine(std::istream&, ...)`: tokenizes one line
  /// (raw bytes, including the trailing `\n`) and returns
  /// `(std::exp(score), label)` pairs, best first.
  pub fn predict_line(
    &self,
    input: &[u8],
    k: i32,
    threshold: f32,
  ) -> Result<Vec<(f32, String)>, FastLangdetectError> {
    if input.is_empty() {
      // `if (in.peek() == EOF) return false;`
      return Ok(Vec::new());
    }
    let mut words = Vec::new();
    let mut labels = Vec::new();
    self.dict.get_line(input, &mut words, &mut labels);
    let preds = self.predict_words(k, &words, threshold)?;
    Ok(
      preds
        .into_iter()
        .map(|(score, lid)| {
          (
            score.exp(),
            // castToPythonString(..., "strict"): labels of a valid model are UTF-8.
            String::from_utf8_lossy(self.dict.get_label(lid)).into_owned(),
          )
        })
        .collect(),
    )
  }

  /// Port of `fasttext.FastText._FastText.predict(text, k, threshold)` for a
  /// single string: returns `(labels, probs)`.
  ///
  /// `threshold` is a Python `float` that pybind11 narrows to `real` (`f32`).
  pub fn predict(
    &self,
    text: &str,
    k: i32,
    threshold: f64,
  ) -> Result<(Vec<String>, Vec<f64>), FastLangdetectError> {
    if text.contains('\n') {
      return Err(FastLangdetectError::InvalidArgument(
        "predict processes one line at a time (remove '\\n')".into(),
      ));
    }
    let mut line = Vec::with_capacity(text.len() + 1);
    line.extend_from_slice(text.as_bytes());
    line.push(b'\n');
    let predictions = self.predict_line(&line, k, threshold as f32)?;
    let mut labels = Vec::with_capacity(predictions.len());
    let mut probs = Vec::with_capacity(predictions.len());
    for (p, l) in predictions {
      probs.push(f64::from(p));
      labels.push(l);
    }
    Ok((labels, probs))
  }
}
