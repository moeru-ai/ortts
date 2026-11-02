//! Port of `ToJyutping/Jyutping.py` (the integer-id constructor and the
//! `jyutping` attribute only).

/// Port of `ToJyutping.Jyutping.onset`.
pub const ONSET: [&str; 20] = [
  "", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "ng", "gw", "kw", "w", "h", "z", "c", "s",
  "j",
];

/// Port of `ToJyutping.Jyutping.nucleus`.
pub const NUCLEUS: [&str; 6] = ["aa", "a", "e", "i", "o", "u"];

/// Port of `ToJyutping.Jyutping.rhyme`.
pub const RHYME: [&str; 13] = [
  "oe", "oen", "oeng", "oet", "oek", "eoi", "eon", "eot", "yu", "yun", "yut", "m", "ng",
];

/// Port of `ToJyutping.Jyutping.coda`.
pub const CODA: [&str; 9] = ["", "i", "u", "m", "n", "ng", "p", "t", "k"];

/// Port of `ToJyutping.Jyutping.Jyutping`, built from an integer id
/// (`Jyutping(x: int)`). The string constructor is left out because it is only
/// used by `customize`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Jyutping {
  /// Port of `Jyutping.id`.
  pub id: u32,
  /// Port of `Jyutping.onset_id`.
  pub onset_id: u32,
  /// Port of `Jyutping.onset`.
  pub onset: &'static str,
  /// Port of `Jyutping.rhyme_id`.
  pub rhyme_id: u32,
  /// Port of `Jyutping.rhyme`.
  pub rhyme: String,
  /// Port of `Jyutping.tone_id`.
  pub tone_id: u32,
  /// Port of `Jyutping.tone`.
  pub tone: String,
  /// Port of `Jyutping.jyutping`.
  pub jyutping: String,
}

impl Jyutping {
  /// Port of `Jyutping.__init__` with an `int` argument.
  pub fn new(x: u32) -> Self {
    let onset_id = x / 402;
    let onset = ONSET[onset_id as usize];
    let rhyme_id = (x % 402) / 6;
    let rhyme = if rhyme_id >= 54 {
      RHYME[(rhyme_id - 54) as usize].to_string()
    } else {
      format!(
        "{}{}",
        NUCLEUS[(rhyme_id / 9) as usize],
        CODA[(rhyme_id % 9) as usize]
      )
    };
    let tone_id = x % 6;
    let tone = (tone_id + 1).to_string();
    let jyutping = format!("{onset}{rhyme}{tone}");
    Self {
      id: x,
      onset_id,
      onset,
      rhyme_id,
      rhyme,
      tone_id,
      tone,
      jyutping,
    }
  }
}

/// Port of `ToJyutping.Jyutping.JyutpingList`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct JyutpingList(pub Vec<Jyutping>);

impl JyutpingList {
  /// Port of the `JyutpingList.jyutping` property: syllables joined by a space.
  pub fn jyutping(&self) -> String {
    self
      .0
      .iter()
      .map(|j| j.jyutping.as_str())
      .collect::<Vec<_>>()
      .join(" ")
  }
}

/// The Python value type `Union[Jyutping, JyutpingList]` stored in trie nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JyutpingValue {
  /// A single `Jyutping`.
  Jyutping(Jyutping),
  /// A `JyutpingList`.
  JyutpingList(JyutpingList),
}

impl JyutpingValue {
  /// `getattr(value, 'jyutping')` on either variant.
  pub fn jyutping(&self) -> String {
    match self {
      Self::Jyutping(j) => j.jyutping.clone(),
      Self::JyutpingList(l) => l.jyutping(),
    }
  }

  /// `value[i]` (`__getitem__`) as used by `Trie.get` for multi-character matches.
  ///
  /// # Panics
  ///
  /// Panics like Python would (`TypeError` / `IndexError`) when the value is a
  /// single `Jyutping` or the index is out of range. The bundled data never
  /// triggers this.
  pub fn getitem(&self, i: usize) -> &Jyutping {
    match self {
      Self::Jyutping(_) => panic!("'Jyutping' object is not subscriptable"),
      Self::JyutpingList(l) => l.0.get(i).expect("list index out of range"),
    }
  }
}
