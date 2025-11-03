use jieba_rs::Jieba;
use ortts_shared::{AppError, Downloader};
use std::collections::HashMap;
use unicode_general_category::{GeneralCategory, get_general_category};

pub struct ChineseCangjieConverter {
  word2cj: HashMap<String, String>,
  cj2word: HashMap<String, Vec<String>>,
  jieba: Jieba,
}

impl ChineseCangjieConverter {
  pub async fn new() -> Result<Self, AppError> {
    let mut word2cj = HashMap::<String, String>::new();
    let mut cj2word = HashMap::<String, Vec<String>>::new();

    let json = Downloader::new()
      .get_str(
        "onnx-community/chatterbox-multilingual-ONNX",
        "Cangjie5_TC.json",
      )
      .await?;

    let data: Vec<String> = serde_json::from_str(&json)?;

    for entry in data {
      let parts: Vec<&str> = entry.split('\t').collect();
      if parts.len() >= 2 {
        let word = parts[0].to_string();
        let code = parts[1].to_string();

        word2cj.insert(word.clone(), code.clone());
        cj2word.entry(code).or_default().push(word);
      }
    }

    Ok(Self {
      cj2word,
      word2cj,
      jieba: Jieba::new(),
    })
  }

  fn cangjie_encode(&self, normed_glyph: &str) -> Option<String> {
    // fn _cangjie_encode(&self, glyph: &str) -> Option<String> {
    // let normed_glyph = glyph;

    let code = self.word2cj.get(normed_glyph)?;

    let words_for_code = self.cj2word.get(code)?;
    let index = words_for_code
      .iter()
      .position(|w| w == normed_glyph)
      .unwrap_or(0);

    let index_str = if index > 0 {
      index.to_string()
    } else {
      String::new()
    };

    Some(format!("{code}{index_str}"))
  }

  pub fn convert(&self, text: &str) -> String {
    let full_text = self.jieba.cut(text, false).join(" ");

    let mut output = Vec::new();

    for t in full_text.chars() {
      let t_str = t.to_string();

      let is_chinese_glyph = get_general_category(t) == GeneralCategory::OtherLetter;

      if is_chinese_glyph {
        if let Some(cangjie) = self.cangjie_encode(&t_str) {
          let mut code: Vec<String> = Vec::new();
          for c in cangjie.chars() {
            code.push(format!("[cj_{c}]"));
          }
          code.push("[cj_.]".to_string());
          output.push(code.join(""));
        } else {
          output.push(t_str);
        }
      } else {
        output.push(t_str);
      }
    }

    output.join("")
  }
}
