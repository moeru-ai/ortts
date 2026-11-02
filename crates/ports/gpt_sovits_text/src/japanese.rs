//! Port of `GPT_SoVITS/text/japanese.py` (GPT-SoVITS @ 04a44eb).
//!
//! Upstream calls `pyopenjtalk` (Open JTalk 1.11 with `open_jtalk_dic_utf_8-1.11`).
//! This port uses `jpreprocess` (a Rust rewrite of Open JTalk's frontend) with
//! its bundled `naist-jdic`. The [`pyopenjtalk`] submodule wraps `jpreprocess`
//! behind the pyopenjtalk functions that `japanese.py` calls. See the crate
//! README and `parity/japanese/README.md` for the measured
//! differences between the two.

use std::sync::LazyLock;

use regex::Regex;

/// Port of the `pyopenjtalk` functions that `japanese.py` calls, backed by
/// `jpreprocess`.
///
/// `pyopenjtalk` keeps one global `OpenJTalk` instance. Upstream replaces it at
/// import time with one that also loads `ja_userdic/user.dict` (compiled from
/// `ja_userdic/userdict.csv`). `global_jtalk` is the equivalent: one lazily
/// built `jpreprocess` instance with the bundled `naist-jdic` and the user
/// dictionary built from the embedded `userdict.csv`.
pub mod pyopenjtalk {
  use std::sync::LazyLock;

  use anyhow::Result;
  use jpreprocess::kind::JPreprocessDictionaryKind;
  use jpreprocess::{Dictionary, JPreprocess, SystemDictionaryConfig, UserDictionary};
  use jpreprocess_core::JPreprocessResult;
  use jpreprocess_core::error::DictionaryError;
  use jpreprocess_core::token::{Token, Tokenizer};
  use jpreprocess_core::word_entry::WordEntry;
  use jpreprocess_dictionary::dictionary::to_dict::JPreprocessDictionaryBuilder;
  use jpreprocess_dictionary::dictionary::word_encoding::JPreprocessDictionaryWordEncoding;
  use jpreprocess_dictionary::word_data::get_word_data;
  pub use jpreprocess_njd::NJDNode;
  use lindera_dictionary::dictionary::prefix_dictionary::PrefixDictionary;

  /// `GPT_SoVITS/text/ja_userdic/userdict.csv`, the user dictionary in Open JTalk
  /// (MeCab) CSV format.
  pub const USERDIC_CSV: &str = include_str!("../data/ja_userdic/userdict.csv");

  /// Port of MeCab's `tokenizeCSV(str, out, max)` (`mecab/src/utils.cpp`), the
  /// CSV reader of `mecab-dict-index`. Leading spaces and tabs of a field are
  /// skipped, a field may be quoted with `"` (`""` inside quotes is a literal
  /// `"`), and the text after a closing quote up to the next `,` is dropped.
  /// The `max`-th field is the raw rest of the line.
  pub fn tokenize_csv(line: &str, max: usize) -> Vec<String> {
    let bytes = line.as_bytes();
    let eos = bytes.len();
    let find_comma = |from: usize| {
      bytes[from..]
        .iter()
        .position(|&b| b == b',')
        .map_or(eos, |p| from + p)
    };
    let mut out = Vec::new();
    let mut max = max;
    let mut i = 0;
    while i < eos {
      while i < eos && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
      }
      let field;
      if max <= 1 {
        // The last field is not terminated: it is the rest of the line.
        field = line[i..].to_string();
        out.push(field);
        break;
      }
      if i < eos && bytes[i] == b'"' {
        i += 1;
        let mut buf = Vec::new();
        while i < eos {
          if bytes[i] == b'"' {
            i += 1;
            if i >= eos || bytes[i] != b'"' {
              break;
            }
          }
          buf.push(bytes[i]);
          i += 1;
        }
        i = find_comma(i.min(eos));
        field = String::from_utf8(buf).expect("utf-8 field");
      } else {
        let end = find_comma(i);
        field = line[i..end].to_string();
        i = end;
      }
      out.push(field);
      max -= 1;
      // `++str` of the for loop skips the separator.
      i += 1;
    }
    out
  }

  /// Port of `pyopenjtalk.mecab_dict_index(path, out_path)`: compiles a user
  /// dictionary CSV (Open JTalk format) into a `jpreprocess` user dictionary,
  /// kept in memory instead of being written to `out_path`.
  ///
  /// `mecab-dict-index` reads `surface,left_id,right_id,cost` with
  /// [`tokenize_csv`] and keeps the rest of the line as the raw feature string,
  /// which Open JTalk later splits on `,` without unquoting. This does the same.
  pub fn mecab_dict_index(csv: &str) -> Result<UserDictionary> {
    let rows: Vec<Vec<String>> = csv
      .lines()
      .filter(|line| !line.is_empty())
      .map(|line| {
        let mut cols = tokenize_csv(line, 5);
        let feature = cols.pop().unwrap_or_default();
        cols.extend(feature.split(',').map(str::to_string));
        jpcommon_compatible_row(&mut cols);
        cols
      })
      .collect();
    let data = rows
      .iter()
      .map(|row| row.iter().map(String::as_str).collect())
      .collect();
    Ok(JPreprocessDictionaryBuilder::default().build_user_dict_from_data(data)?)
  }

  /// Column of `pron` and `acc/mora_size` in an Open JTalk dictionary row.
  const PRON: usize = 12;
  const ACC_MORASIZE: usize = 13;

  /// Whether Open JTalk's label builder (`JPCommonLabel_push_word` with
  /// `jpcommon_mora_list`) accepts `c` in a pronunciation: katakana moras,
  /// the long vowel mark `ー` and the unvoicing mark `’`.
  fn is_jpcommon_pron_char(c: char) -> bool {
    ('\u{30A1}'..='\u{30F4}').contains(&c) || c == 'ー' || c == '’'
  }

  /// Not in upstream: makes a dictionary row behave in `jpreprocess` like
  /// Open JTalk handles it.
  ///
  /// Open JTalk keeps `pron` as a raw string and parses it into moras only
  /// when it builds labels (`JPCommonLabel_push_word`). That parser knows only
  /// katakana moras and stops at the first other character (`バーベキューど`
  /// gives `バーベキュー`, `アドル・タバタバ` gives `アドル`). `jpreprocess`
  /// parses `pron` when the dictionary is built, accepts hiragana and
  /// fullwidth letters as moras, rejects other characters, and rejects a
  /// `mora_size` that differs from the parsed moras. So `pron` is cut at the
  /// first character the Open JTalk label builder rejects, and a `mora_size`
  /// that then no longer matches is dropped (the accent is kept).
  fn jpcommon_compatible_row(cols: &mut [String]) {
    use jpreprocess_core::pronunciation::Pronunciation;

    if cols.len() != 15 || cols[PRON] == "*" {
      return;
    }
    let pron = &cols[PRON];
    let end = pron
      .char_indices()
      .find(|&(_, c)| !is_jpcommon_pron_char(c))
      .map_or(pron.len(), |(i, _)| i);
    let prefix = pron[..end].to_string();
    let mora_size = Pronunciation::parse(&prefix, 0).map_or(0, |p| p.mora_size());
    cols[PRON] = prefix;
    let acc_morasize = cols[ACC_MORASIZE].clone();
    if let Some((acc, size)) = acc_morasize.split_once('/')
      && size.parse::<usize>().ok() != Some(mora_size)
    {
      cols[ACC_MORASIZE] = acc.to_string();
    }
  }

  /// The global `OpenJTalk` instance of `pyopenjtalk` (`_global_jtalk`) after
  /// `japanese.py` called `update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)`.
  pub fn global_jtalk() -> &'static JPreprocess<MecabTokenizer> {
    static JTALK: LazyLock<JPreprocess<MecabTokenizer>> = LazyLock::new(|| {
      let system = SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic)
        .load()
        .expect("load bundled naist-jdic");
      let user = mecab_dict_index(USERDIC_CSV).expect("build ja_userdic/userdict.csv");
      JPreprocess::from_tokenizer(MecabTokenizer::new(system, Some(user)))
    });
    &JTALK
  }

  /// MeCab's `max-grouping-size` default (`mecab/src/tokenizer.cpp`).
  const MAX_GROUPING_SIZE: usize = 24;

  /// Port of MeCab's `CharInfo` (`mecab/src/char_property.h`): the categories
  /// of a character, its first (default) category, and the `invoke`,
  /// `group` and `length` of the default category.
  #[derive(Clone, Copy)]
  struct CharInfo {
    types: u64,
    default_type: usize,
    invoke: bool,
    group: bool,
    length: u32,
  }

  impl CharInfo {
    /// `CharInfo::isKindOf`.
    fn is_kind_of(&self, other: &CharInfo) -> bool {
      self.types & other.types != 0
    }
  }

  /// Which dictionary a lattice node comes from.
  #[derive(Clone, Copy, PartialEq)]
  enum Lex {
    Bos,
    System,
    User,
    Unknown,
  }

  /// Port of MeCab's lattice `Node` (the fields Viterbi needs).
  struct Node {
    lex: Lex,
    entry: lindera_dictionary::viterbi::WordEntry,
    /// Byte range of the surface (leading spaces excluded).
    begin: usize,
    end: usize,
    cost: i64,
    prev: usize,
  }

  /// Not in upstream: a MeCab-compatible tokenizer for `jpreprocess`.
  ///
  /// Open JTalk segments text with MeCab. `jpreprocess` normally uses lindera,
  /// whose Viterbi differs from MeCab in ways that change the segmentation:
  /// it keeps the first of two equally cheap paths where MeCab keeps the
  /// last one (MeCab prepends nodes to `end_node_list`), it does not add
  /// unknown-word candidates inside an unknown word it already added, and
  /// it ignores the `length` column of `char.def`. Ties are common with
  /// `userdict.csv`, whose costs depend only on the surface length. This
  /// is a port of MeCab's `Tokenizer::lookup` and `Viterbi::viterbi`
  /// (`mecab/src/tokenizer.cpp`, `mecab/src/viterbi.cc`) over the lindera
  /// dictionaries.
  pub struct MecabTokenizer {
    system: Dictionary,
    user: Option<UserDictionary>,
  }

  impl MecabTokenizer {
    pub fn new(system: Dictionary, user: Option<UserDictionary>) -> Self {
      Self { system, user }
    }

    /// `CharProperty::getCharInfo`.
    fn char_info(&self, c: char) -> CharInfo {
      let chardef = &self.system.character_definition;
      let categories = chardef.lookup_categories(c);
      let types = categories.iter().fold(0u64, |acc, id| acc | (1 << id.0));
      let default_type = categories.first().map_or(0, |id| id.0);
      let def = &chardef.category_definitions[default_type];
      CharInfo {
        types,
        default_type,
        invoke: def.invoke,
        group: def.group,
        length: def.length,
      }
    }

    /// `CharProperty::seekToOtherType`: advances while each character
    /// shares a category with the previous one. Returns the end position
    /// and the number of characters passed.
    fn seek_to_other_type(&self, text: &str, mut pos: usize, mut c: CharInfo) -> (usize, usize) {
      let mut clen = 0;
      while let Some(ch) = text[pos..].chars().next() {
        let fail = self.char_info(ch);
        if !c.is_kind_of(&fail) {
          break;
        }
        pos += ch.len_utf8();
        clen += 1;
        c = fail;
      }
      (pos, clen)
    }

    /// Dictionary candidates starting at `begin`, in MeCab's insertion
    /// order: system dictionary first, then the user dictionary, shorter
    /// surfaces first, and dictionary order within a surface.
    fn push_dictionary_nodes(
      dict: &PrefixDictionary,
      lex: Lex,
      text: &str,
      begin: usize,
      out: &mut Vec<Node>,
    ) {
      let mut matches: Vec<(usize, lindera_dictionary::viterbi::WordEntry)> =
        dict.prefix(&text[begin..]).collect();
      // jpreprocess stores the entries of one surface in reverse; the word
      // id is the row number of the CSV (stable-sorted by surface).
      matches.sort_by_key(|(len, entry)| (*len, entry.word_id.id));
      for (len, entry) in matches {
        out.push(Node {
          lex,
          entry,
          begin,
          end: begin + len,
          cost: 0,
          prev: 0,
        });
      }
    }

    /// Port of `Tokenizer::lookup`: the candidate nodes starting at `pos`,
    /// in insertion order (MeCab links them in reverse through `bnext`).
    fn lookup(&self, text: &str, pos: usize) -> Vec<Node> {
      let mut out = Vec::new();
      // Skip white space (`space_` is the char info of U+0020).
      let (begin2, _) = self.seek_to_other_type(text, pos, self.char_info(' '));
      let Some(first) = text[begin2..].chars().next() else {
        return out;
      };
      let cinfo = self.char_info(first);

      Self::push_dictionary_nodes(
        &self.system.prefix_dictionary,
        Lex::System,
        text,
        begin2,
        &mut out,
      );
      if let Some(user) = &self.user {
        Self::push_dictionary_nodes(&user.dict, Lex::User, text, begin2, &mut out);
      }
      if !out.is_empty() && !cinfo.invoke {
        return out;
      }

      let unknown = &self.system.unknown_dictionary;
      let add_unknown = |out: &mut Vec<Node>, end: usize| {
        for &id in unknown.lookup_word_ids(
          lindera_dictionary::dictionary::character_definition::CategoryId(cinfo.default_type),
        ) {
          let mut entry = unknown.word_entry(id);
          entry.word_id.id = id;
          out.push(Node {
            lex: Lex::Unknown,
            entry,
            begin: begin2,
            end,
            cost: 0,
            prev: 0,
          });
        }
      };

      let mut begin3 = begin2 + first.len_utf8();
      let mut group_begin3 = None;
      if cinfo.group {
        let tmp = begin3;
        let (end, clen) = self.seek_to_other_type(text, begin3, cinfo);
        if clen <= MAX_GROUPING_SIZE {
          add_unknown(&mut out, end);
        }
        group_begin3 = Some(end);
        begin3 = tmp;
      }
      for _ in 1..=cinfo.length {
        if begin3 > text.len() {
          break;
        }
        if Some(begin3) == group_begin3 {
          continue;
        }
        add_unknown(&mut out, begin3);
        let Some(next) = text[begin3..].chars().next() else {
          break;
        };
        if !cinfo.is_kind_of(&self.char_info(next)) {
          break;
        }
        begin3 += next.len_utf8();
      }
      if out.is_empty() {
        add_unknown(&mut out, begin3);
      }
      out
    }

    /// The surface and `jpreprocess` word entry of a lattice node.
    fn word_entry(&self, node: &Node) -> JPreprocessResult<WordEntry> {
      let dict = match node.lex {
        Lex::System => &self.system.prefix_dictionary,
        Lex::User => &self.user.as_ref().expect("user dictionary").dict,
        Lex::Unknown => {
          let mut details = self
            .system
            .unknown_word_details(node.entry.word_id.id as usize);
          details.resize(12, "");
          return WordEntry::load(&details);
        }
        Lex::Bos => unreachable!("BOS has no word entry"),
      };
      let data = get_word_data(
        &dict.words_idx_data,
        &dict.words_data,
        Some(node.entry.word_id.id as usize),
      )
      .ok_or(DictionaryError::IdNotFound(node.entry.word_id.id))?;
      Ok(JPreprocessDictionaryWordEncoding::deserialize(data).map_err(DictionaryError::from)?)
    }
  }

  impl Tokenizer for MecabTokenizer {
    /// Port of `Viterbi::viterbi` and `Viterbi::connect`.
    fn tokenize<'a>(&'a self, text: &'a str) -> JPreprocessResult<Vec<impl 'a + Token>> {
      let matrix = &self.system.connection_cost_matrix;
      let len = text.len();
      let mut nodes = vec![Node {
        lex: Lex::Bos,
        entry: Default::default(),
        begin: 0,
        end: 0,
        cost: 0,
        prev: 0,
      }];
      // MeCab's `end_node_list`, newest node last (MeCab iterates newest first).
      let mut end_nodes: Vec<Vec<usize>> = vec![Vec::new(); len + 1];
      end_nodes[0].push(0);

      let connect =
        |nodes: &mut Vec<Node>, end_nodes: &mut Vec<Vec<usize>>, pos: usize, mut rnode: Node| {
          let mut best: Option<(usize, i64)> = None;
          for &l in end_nodes[pos].iter().rev() {
            let lnode = &nodes[l];
            let cost = lnode.cost
              + matrix.cost(lnode.entry.right_id(), rnode.entry.left_id()) as i64
              + rnode.entry.word_cost as i64;
            if best.is_none_or(|(_, b)| cost < b) {
              best = Some((l, cost));
            }
          }
          let (prev, cost) = best.expect("lattice node has a left node");
          rnode.prev = prev;
          rnode.cost = cost;
          let end = rnode.end;
          nodes.push(rnode);
          end_nodes[end].push(nodes.len() - 1);
        };

      for pos in 0..len {
        if end_nodes[pos].is_empty() {
          continue;
        }
        for rnode in self.lookup(text, pos).into_iter().rev() {
          connect(&mut nodes, &mut end_nodes, pos, rnode);
        }
      }

      let Some(eos_pos) = (0..=len).rev().find(|&p| !end_nodes[p].is_empty()) else {
        return Ok(Vec::new());
      };
      connect(
        &mut nodes,
        &mut end_nodes,
        eos_pos,
        Node {
          lex: Lex::Bos,
          entry: Default::default(),
          begin: eos_pos,
          end: eos_pos,
          cost: 0,
          prev: 0,
        },
      );

      let mut path = Vec::new();
      let mut i = nodes[nodes.len() - 1].prev;
      while i != 0 {
        path.push(i);
        i = nodes[i].prev;
      }
      path.reverse();
      path
        .into_iter()
        .map(|i| {
          let node = &nodes[i];
          Ok((
            text[node.begin..node.end].to_string(),
            self.word_entry(node)?,
          ))
        })
        .collect::<JPreprocessResult<Vec<(String, WordEntry)>>>()
    }
  }

  /// Port of `pyopenjtalk.run_frontend(text)`: the NJD nodes after Open
  /// JTalk's text processing. `NJDNode::to_string` gives the feature in the
  /// layout
  /// `string,pos,pos_group1,pos_group2,pos_group3,ctype,cform,orig,read,pron,acc/mora_size,chain_rule,chain_flag`,
  /// where `orig` is the surface string because `jpreprocess` drops the base
  /// form.
  pub fn run_frontend(text: &str) -> Result<Vec<NJDNode>> {
    let mut njd = global_jtalk().text_to_njd(text)?;
    njd.preprocess();
    Ok(njd.nodes)
  }

  /// Port of `pyopenjtalk.make_label(njd_features)`: the full-context labels
  /// as strings.
  pub fn make_label(njd_features: &[NJDNode]) -> Vec<String> {
    jpreprocess_jpcommon::njdnodes_to_features(&jpcommon_long_vowels(njd_features))
      .into_iter()
      .map(|label| label.to_string())
      .collect()
  }

  /// Not in upstream: moves word-initial long vowel marks (`ー`) the way Open
  /// JTalk's `JPCommonLabel_push_word` does, before `jpreprocess` builds the
  /// labels.
  ///
  /// Open JTalk appends a `ー` that starts a word to the previous word (its
  /// phoneme and accent phrase), and drops it when no phoneme was pushed yet
  /// or a pause (`、`, `？`) is pending. Only the next non-`ー` mora starts
  /// the new word. `jpreprocess` starts the new word at the `ー` and copies
  /// the previous phoneme even when that is `sil` or `pau`.
  fn jpcommon_long_vowels(nodes: &[NJDNode]) -> Vec<NJDNode> {
    use jpreprocess_core::pronunciation::{MoraEnum, Pronunciation};

    let mut out: Vec<NJDNode> = Vec::with_capacity(nodes.len());
    // Index in `out` of the last word with moras since the last pause.
    let mut last_word = None;
    for node in nodes {
      let pron = node.get_pron();
      if pron.is_touten() || pron.is_question() {
        last_word = None;
        out.push(node.clone());
        continue;
      }
      let moras = pron.moras();
      let lead = moras
        .iter()
        .take_while(|m| matches!(m.mora_enum, MoraEnum::Long))
        .count();
      if lead > 0 {
        if let Some(i) = last_word {
          let long = Pronunciation::new(moras[..lead].to_vec(), 0);
          let prev: &mut NJDNode = &mut out[i];
          prev.get_pron_mut().transfer_from(&long);
        }
        if lead == moras.len() {
          continue;
        }
        let mut node = node.clone();
        node.set_pron(Pronunciation::new(moras[lead..].to_vec(), pron.accent()));
        last_word = Some(out.len());
        out.push(node);
        continue;
      }
      if !moras.is_empty() {
        last_word = Some(out.len());
      }
      out.push(node.clone());
    }
    out
  }

  /// Port of `pyopenjtalk.g2p(text)` (`OpenJTalk.g2p` with `kana=False`,
  /// `join=True`): the current phoneme of every label but the first and the
  /// last, joined with spaces.
  pub fn g2p(text: &str) -> Result<String> {
    let labels = make_label(&run_frontend(text)?);
    let inner = if labels.len() >= 2 {
      &labels[1..labels.len() - 1]
    } else {
      &labels[..0]
    };
    let prons: Vec<&str> = inner
      .iter()
      .map(|s| {
        // s.split("-")[1].split("+")[0]
        let after = s.split('-').nth(1).unwrap_or_default();
        after.split('+').next().unwrap_or_default()
      })
      .collect();
    Ok(prons.join(" "))
  }
}

/// Private copy of `punctuation` from `GPT_SoVITS/text/symbols.py`
/// (`["!", "?", "…", ",", "."]` plus the appended `"-"`). The `symbols` module
/// is ported separately.
const PUNCTUATION: [&str; 6] = ["!", "?", "…", ",", ".", "-"];

/// Port of `_japanese_characters`: matches Japanese without punctuation marks.
static JAPANESE_CHARACTERS: LazyLock<Regex> = LazyLock::new(|| {
  Regex::new(
        r"^[A-Za-z\d\x{3005}\x{3040}-\x{30ff}\x{4e00}-\x{9fff}\x{ff11}-\x{ff19}\x{ff21}-\x{ff3a}\x{ff41}-\x{ff5a}\x{ff66}-\x{ff9d}]",
    )
    .unwrap()
});

/// Port of `_japanese_marks`: matches non-Japanese characters or punctuation marks.
static JAPANESE_MARKS: LazyLock<Regex> = LazyLock::new(|| {
  Regex::new(
        r"[^A-Za-z\d\x{3005}\x{3040}-\x{30ff}\x{4e00}-\x{9fff}\x{ff11}-\x{ff19}\x{ff21}-\x{ff3a}\x{ff41}-\x{ff5a}\x{ff66}-\x{ff9d}]",
    )
    .unwrap()
});

/// Port of `_symbols_to_japanese`: list of (symbol, Japanese) pairs for marks.
static SYMBOLS_TO_JAPANESE: LazyLock<Vec<(Regex, &'static str)>> =
  LazyLock::new(|| vec![(Regex::new("％").unwrap(), "パーセント")]);

/// Port of `post_replace_ph(ph)`.
pub fn post_replace_ph(ph: &str) -> String {
  let rep = match ph {
    "：" => ",",
    "；" => ",",
    "，" => ",",
    "。" => ".",
    "！" => "!",
    "？" => "?",
    "\n" => ".",
    "·" => ",",
    "、" => ",",
    "..." => "…",
    _ => ph,
  };
  rep.to_string()
}

/// Port of `replace_consecutive_punctuation(text)`.
pub fn replace_consecutive_punctuation(text: &str) -> String {
  static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let punctuations: String = PUNCTUATION.iter().map(|p| regex::escape(p)).collect();
    Regex::new(&format!("([{punctuations}])([{punctuations}])+")).unwrap()
  });
  PATTERN.replace_all(text, "${1}").into_owned()
}

/// Port of `symbols_to_japanese(text)`.
pub fn symbols_to_japanese(text: &str) -> String {
  let mut text = text.to_string();
  for (regex, replacement) in SYMBOLS_TO_JAPANESE.iter() {
    text = regex.replace_all(&text, *replacement).into_owned();
  }
  text
}

/// Port of `preprocess_jap(text, with_prosody=False)`.
pub fn preprocess_jap(text: &str, with_prosody: bool) -> anyhow::Result<Vec<String>> {
  let text = symbols_to_japanese(text);
  // English words to lower case, should have no influence on japanese words.
  let text = text.to_lowercase();
  let sentences: Vec<&str> = JAPANESE_MARKS.split(&text).collect();
  let marks: Vec<&str> = JAPANESE_MARKS
    .find_iter(&text)
    .map(|m| m.as_str())
    .collect();
  let mut out = Vec::new();
  for (i, sentence) in sentences.iter().enumerate() {
    if JAPANESE_CHARACTERS.is_match(sentence) {
      if with_prosody {
        let phones = pyopenjtalk_g2p_prosody(sentence, true)?;
        if phones.len() >= 2 {
          out.extend_from_slice(&phones[1..phones.len() - 1]);
        }
      } else {
        let p = pyopenjtalk::g2p(sentence)?;
        out.extend(p.split(' ').map(str::to_string));
      }
    }

    if i < marks.len() {
      if marks[i] == " " {
        // 防止意外的UNK
        continue;
      }
      out.push(marks[i].replace(' ', ""));
    }
  }
  Ok(out)
}

/// Port of `text_normalize(text)`.
pub fn text_normalize(text: &str) -> String {
  // todo: jap text normalize

  // 避免重复标点引起的参考泄露
  replace_consecutive_punctuation(text)
}

/// Port of `pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)` (copied
/// upstream from espnet): phoneme + prosody symbol sequence from full-context
/// labels.
pub fn pyopenjtalk_g2p_prosody(
  text: &str,
  drop_unvoiced_vowels: bool,
) -> anyhow::Result<Vec<String>> {
  static P3: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\-(.*?)\+").unwrap());
  static E3: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"!(\d+)_").unwrap());
  static A1: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/A:([0-9\-]+)\+").unwrap());
  static A2: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\+(\d+)\+").unwrap());
  static A3: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\+(\d+)/").unwrap());
  static F1: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/F:(\d+)_").unwrap());
  let labels = pyopenjtalk::make_label(&pyopenjtalk::run_frontend(text)?);
  let n_labels = labels.len();

  let mut phones = Vec::new();
  for n in 0..n_labels {
    let lab_curr = &labels[n];

    // current phoneme
    let mut p3 = P3
      .captures(lab_curr)
      .expect("label has a current phoneme")
      .get(1)
      .unwrap()
      .as_str()
      .to_string();
    // deal unvoiced vowels as normal vowels
    if drop_unvoiced_vowels && "AEIOU".contains(p3.as_str()) {
      p3 = p3.to_lowercase();
    }

    // deal with sil at the beginning and the end of text
    if p3 == "sil" {
      anyhow::ensure!(n == 0 || n == n_labels - 1, "AssertionError");
      if n == 0 {
        phones.push("^".to_string());
      } else if n == n_labels - 1 {
        // check question form or not
        let e3 = _numeric_feature_by_regex(&E3, lab_curr);
        if e3 == 0 {
          phones.push("$".to_string());
        } else if e3 == 1 {
          phones.push("?".to_string());
        }
      }
      continue;
    } else if p3 == "pau" {
      phones.push("_".to_string());
      continue;
    } else {
      phones.push(p3.clone());
    }

    // accent type and position info (forward or backward)
    let a1 = _numeric_feature_by_regex(&A1, lab_curr);
    let a2 = _numeric_feature_by_regex(&A2, lab_curr);
    let a3 = _numeric_feature_by_regex(&A3, lab_curr);

    // number of mora in accent phrase
    let f1 = _numeric_feature_by_regex(&F1, lab_curr);

    // Python raises IndexError when the last label is not `sil`.
    let Some(lab_next) = labels.get(n + 1) else {
      anyhow::bail!("IndexError");
    };
    let a2_next = _numeric_feature_by_regex(&A2, lab_next);
    // accent phrase border
    if a3 == 1 && a2_next == 1 && "aeiouAEIOUNcl".contains(p3.as_str()) {
      phones.push("#".to_string());
    }
    // pitch falling
    else if a1 == 0 && a2_next == a2 + 1 && a2 != f1 {
      phones.push("]".to_string());
    }
    // pitch rising
    else if a2 == 1 && a2_next == 2 {
      phones.push("[".to_string());
    }
  }

  Ok(phones)
}

/// Port of `_numeric_feature_by_regex(regex, s)` (copied upstream from espnet).
pub fn _numeric_feature_by_regex(regex: &Regex, s: &str) -> i64 {
  match regex.captures(s) {
    None => -50,
    Some(caps) => caps[1].parse().expect("integer feature"),
  }
}

/// Port of `g2p(norm_text, with_prosody=True)`.
pub fn g2p(norm_text: &str, with_prosody: bool) -> anyhow::Result<Vec<String>> {
  let phones = preprocess_jap(norm_text, with_prosody)?;
  let phones = phones.iter().map(|i| post_replace_ph(i)).collect();
  // todo: implement tones and word2ph
  Ok(phones)
}
