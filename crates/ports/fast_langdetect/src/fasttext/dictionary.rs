//! Port of fastText `src/dictionary.{h,cc}` (loading and `getLine` for
//! supervised prediction).

use std::collections::HashMap;
use std::io::Read;

use super::args::Args;
use super::read::{read_i8, read_i32, read_i64};

/// Port of `Dictionary::EOS`.
pub const EOS: &[u8] = b"</s>";
/// Port of `Dictionary::BOW`.
pub const BOW: &[u8] = b"<";
/// Port of `Dictionary::EOW`.
pub const EOW: &[u8] = b">";

/// Port of fastText `enum class entry_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryType {
  /// `entry_type::word` (0).
  Word,
  /// `entry_type::label` (1).
  Label,
}

/// Port of fastText `struct entry` (without the precomputed `subwords`, which
/// [`Dictionary::get_subwords`] computes on demand with the same result).
#[derive(Debug, Clone)]
pub struct Entry {
  /// `entry::word` (raw bytes; fastText does not validate UTF-8).
  pub word: Vec<u8>,
  /// `entry::count`.
  pub count: i64,
  /// `entry::type`.
  pub entry_type: EntryType,
}

/// Port of fastText `class Dictionary`.
#[derive(Debug, Clone)]
pub struct Dictionary {
  args: Args,
  /// Replaces the open-addressing `word2int_` table: `find` there returns
  /// the slot of the exact word, so a hash map gives the same ids.
  word2int: HashMap<Vec<u8>, i32>,
  words: Vec<Entry>,
  size: i32,
  nwords: i32,
  nlabels: i32,
  ntokens: i64,
  pruneidx_size: i64,
  pruneidx: HashMap<i32, i32>,
}

/// Port of `Dictionary::hash(const std::string&)`: FNV-1a over the bytes, with
/// each byte sign-extended through `int8_t` before the xor.
pub fn hash(s: &[u8]) -> u32 {
  let mut h: u32 = 2166136261;
  for &b in s {
    h ^= (b as i8) as i32 as u32;
    h = h.wrapping_mul(16777619);
  }
  h
}

impl Dictionary {
  /// Port of `Dictionary::Dictionary(std::shared_ptr<Args>, std::istream&)`
  /// (`Dictionary::load`).
  pub fn load<R: Read>(args: Args, r: &mut R) -> std::io::Result<Self> {
    let size = read_i32(r)?;
    let nwords = read_i32(r)?;
    let nlabels = read_i32(r)?;
    let ntokens = read_i64(r)?;
    let pruneidx_size = read_i64(r)?;
    let mut words = Vec::with_capacity(size.max(0) as usize);
    let mut byte = [0u8; 1];
    for _ in 0..size {
      let mut word = Vec::new();
      loop {
        r.read_exact(&mut byte)?;
        if byte[0] == 0 {
          break;
        }
        word.push(byte[0]);
      }
      let count = read_i64(r)?;
      let entry_type = if read_i8(r)? == 1 {
        EntryType::Label
      } else {
        EntryType::Word
      };
      words.push(Entry {
        word,
        count,
        entry_type,
      });
    }
    let mut pruneidx = HashMap::new();
    for _ in 0..pruneidx_size.max(0) {
      let first = read_i32(r)?;
      let second = read_i32(r)?;
      pruneidx.insert(first, second);
    }
    let mut word2int = HashMap::with_capacity(words.len());
    for (i, e) in words.iter().enumerate() {
      word2int.insert(e.word.clone(), i as i32);
    }
    Ok(Self {
      args,
      word2int,
      words,
      size,
      nwords,
      nlabels,
      ntokens,
      pruneidx_size,
      pruneidx,
    })
  }

  /// Port of `Dictionary::nwords()`.
  pub fn nwords(&self) -> i32 {
    self.nwords
  }

  /// Port of `Dictionary::nlabels()`.
  pub fn nlabels(&self) -> i32 {
    self.nlabels
  }

  /// Port of `Dictionary::ntokens()`.
  pub fn ntokens(&self) -> i64 {
    self.ntokens
  }

  /// Port of `Dictionary::size_`.
  pub fn size(&self) -> i32 {
    self.size
  }

  /// Port of `Dictionary::isPruned()`.
  pub fn is_pruned(&self) -> bool {
    self.pruneidx_size >= 0
  }

  /// Port of `Dictionary::getId(const std::string&, uint32_t h)`.
  pub fn get_id(&self, w: &[u8]) -> i32 {
    self.word2int.get(w).copied().unwrap_or(-1)
  }

  /// Port of `Dictionary::getType(int32_t id)`.
  pub fn get_type_id(&self, id: i32) -> EntryType {
    self.words[id as usize].entry_type
  }

  /// Port of `Dictionary::getType(const std::string&)`.
  pub fn get_type(&self, w: &[u8]) -> EntryType {
    if w.starts_with(self.args.label.as_bytes()) {
      EntryType::Label
    } else {
      EntryType::Word
    }
  }

  /// Port of `Dictionary::getLabel(int32_t lid)`.
  pub fn get_label(&self, lid: i32) -> &[u8] {
    &self.words[(lid + self.nwords) as usize].word
  }

  /// Port of `Dictionary::getCounts(entry_type)`.
  pub fn get_counts(&self, entry_type: EntryType) -> Vec<i64> {
    self
      .words
      .iter()
      .filter(|e| e.entry_type == entry_type)
      .map(|e| e.count)
      .collect()
  }

  /// Port of `Dictionary::pushHash`.
  fn push_hash(&self, hashes: &mut Vec<i32>, mut id: i32) {
    if self.pruneidx_size == 0 || id < 0 {
      return;
    }
    if self.pruneidx_size > 0 {
      match self.pruneidx.get(&id) {
        Some(&v) => id = v,
        None => return,
      }
    }
    hashes.push(self.nwords + id);
  }

  /// Port of `Dictionary::computeSubwords(const std::string&, std::vector<int32_t>&)`.
  pub fn compute_subwords(&self, word: &[u8], ngrams: &mut Vec<i32>) {
    let maxn = self.args.maxn as usize;
    let minn = self.args.minn as usize;
    for i in 0..word.len() {
      if (word[i] & 0xC0) == 0x80 {
        continue;
      }
      let mut ngram: Vec<u8> = Vec::new();
      let mut j = i;
      let mut n = 1usize;
      while j < word.len() && n <= maxn {
        ngram.push(word[j]);
        j += 1;
        while j < word.len() && (word[j] & 0xC0) == 0x80 {
          ngram.push(word[j]);
          j += 1;
        }
        if n >= minn && !(n == 1 && (i == 0 || j == word.len())) {
          let h = (hash(&ngram) % self.args.bucket as u32) as i32;
          self.push_hash(ngrams, h);
        }
        n += 1;
      }
    }
  }

  /// Port of `Dictionary::getSubwords(int32_t i)`: the word id followed by its
  /// character n-grams (`Dictionary::initNgrams`).
  pub fn get_subwords(&self, i: i32, out: &mut Vec<i32>) {
    out.push(i);
    let w = &self.words[i as usize].word;
    if w.as_slice() != EOS {
      let mut word = Vec::with_capacity(w.len() + 2);
      word.extend_from_slice(BOW);
      word.extend_from_slice(w);
      word.extend_from_slice(EOW);
      self.compute_subwords(&word, out);
    }
  }

  /// Port of `Dictionary::addSubwords`.
  fn add_subwords(&self, line: &mut Vec<i32>, token: &[u8], wid: i32) {
    if wid < 0 {
      if token != EOS {
        let mut word = Vec::with_capacity(token.len() + 2);
        word.extend_from_slice(BOW);
        word.extend_from_slice(token);
        word.extend_from_slice(EOW);
        self.compute_subwords(&word, line);
      }
    } else if self.args.maxn <= 0 {
      line.push(wid);
    } else {
      self.get_subwords(wid, line);
    }
  }

  /// Port of `Dictionary::addWordNgrams`. `hashes` holds `int32_t` values, so
  /// `uint64_t h = hashes[i]` sign-extends them.
  fn add_word_ngrams(&self, line: &mut Vec<i32>, hashes: &[i32], n: i32) {
    let bucket = self.args.bucket as i64 as u64;
    for i in 0..hashes.len() as i32 {
      let mut h = hashes[i as usize] as i64 as u64;
      let mut j = i + 1;
      while (j as usize) < hashes.len() && j < i + n {
        h = h
          .wrapping_mul(116049371)
          .wrapping_add(hashes[j as usize] as i64 as u64);
        self.push_hash(line, (h % bucket) as i32);
        j += 1;
      }
    }
  }

  /// Port of `Dictionary::readWord(std::istream&, std::string&)`. `pos` is
  /// the stream position; `sungetc` is a step back.
  fn read_word(input: &[u8], pos: &mut usize, word: &mut Vec<u8>) -> bool {
    word.clear();
    while *pos < input.len() {
      let c = input[*pos];
      *pos += 1;
      if matches!(c, b' ' | b'\n' | b'\r' | b'\t' | 0x0B | 0x0C | 0) {
        if word.is_empty() {
          if c == b'\n' {
            word.extend_from_slice(EOS);
            return true;
          }
          continue;
        } else {
          if c == b'\n' {
            *pos -= 1;
          }
          return true;
        }
      }
      word.push(c);
    }
    !word.is_empty()
  }

  /// Port of `Dictionary::getLine(std::istream&, std::vector<int32_t>& words,
  /// std::vector<int32_t>& labels)`. Returns the token count.
  pub fn get_line(&self, input: &[u8], words: &mut Vec<i32>, labels: &mut Vec<i32>) -> i32 {
    let mut word_hashes: Vec<i32> = Vec::new();
    let mut token = Vec::new();
    let mut ntokens = 0;
    let mut pos = 0usize;
    words.clear();
    labels.clear();
    while Self::read_word(input, &mut pos, &mut token) {
      let h = hash(&token);
      let wid = self.get_id(&token);
      let entry_type = if wid < 0 {
        self.get_type(&token)
      } else {
        self.get_type_id(wid)
      };
      ntokens += 1;
      if entry_type == EntryType::Word {
        self.add_subwords(words, &token, wid);
        word_hashes.push(h as i32);
      } else if entry_type == EntryType::Label && wid >= 0 {
        labels.push(wid - self.nwords);
      }
      if token == EOS {
        break;
      }
    }
    self.add_word_ngrams(words, &word_hashes, self.args.word_ngrams);
    ntokens
  }
}
