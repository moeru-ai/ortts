//! Port of `ToJyutping/Trie.py` (the module-level `trie.txt` decoder and
//! `Trie.get`).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::jyutping::{Jyutping, JyutpingList, JyutpingValue};

/// The bundled `ToJyutping/trie.txt` from ToJyutping 3.2.0.
pub const TRIE_TXT: &str = include_str!("../data/trie.txt");

/// Port of `ToJyutping.Trie.Node`: a `dict` of child nodes plus the optional
/// value list `v`. The customization map `m` is left out.
#[derive(Debug, Default)]
pub struct Node {
  /// The `dict[str, Node]` part of the Python node.
  pub children: HashMap<char, Node>,
  /// Port of `Node.v`.
  pub v: Option<Vec<JyutpingValue>>,
}

impl Node {
  /// Port of `dict.get` on a node.
  pub fn get(&self, c: char) -> Option<&Node> {
    self.children.get(&c)
  }
}

/// Port of `ToJyutping.Trie.set_default_node`.
pub fn set_default_node(n: &mut Node, c: char) -> &mut Node {
  n.children.entry(c).or_default()
}

/// Port of the module-level decoding loop in `ToJyutping/Trie.py`, which
/// builds `root` from `trie.txt`.
pub fn decode(text: &str) -> Node {
  let s: Vec<char> = text.chars().collect();
  let ord = |c: char| c as u32;
  let mut root = Node::default();
  // `n` is a stack of nodes; Rust keeps it as a stack of key paths from root.
  let mut n: Vec<Vec<char>> = vec![Vec::new()];
  let mut l: Vec<usize> = vec![0];
  let mut i = 1;
  while let Some(base) = n.last() {
    let mut path = base.clone();
    let mut d = *l.last().expect("l and n have the same length");
    while ord(s[i]) >= 256 {
      path.push(s[i]);
      i += 1;
      d += 1;
    }
    let mut v = Vec::new();
    while ord(s[i]) < 123 {
      let mut c = 0;
      let mut w = Vec::new();
      while c < d {
        w.push(Jyutping::new((ord(s[i]) - 33) * 90 + (ord(s[i + 1]) - 33)));
        i += 2;
        if s[i] == '~' {
          i += 1;
        } else {
          c += 1;
        }
      }
      v.push(if w.len() == 1 {
        JyutpingValue::Jyutping(w.pop().expect("len is 1"))
      } else {
        JyutpingValue::JyutpingList(JyutpingList(w))
      });
    }
    let p = path.iter().fold(&mut root, |p, &c| set_default_node(p, c));
    if !v.is_empty() {
      p.v = Some(v);
    }
    if s[i] == '{' {
      i += 1;
      n.push(path);
      l.push(d);
    } else if s[i] == '}' {
      i += 1;
      n.pop();
      l.pop();
    }
  }
  root
}

/// Port of `ToJyutping.Trie.root`, decoded lazily on first use.
pub fn root() -> &'static Node {
  static ROOT: OnceLock<Node> = OnceLock::new();
  ROOT.get_or_init(|| decode(TRIE_TXT))
}

/// Port of `ToJyutping.Trie.Trie`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Trie;

impl Trie {
  /// Port of `Trie.get(s, attr=None)`: greedy longest match from each
  /// position. Each item is one input character and its value.
  pub fn get(&self, s: &str) -> Vec<(String, Option<JyutpingValue>)> {
    let s: Vec<char> = s.chars().collect();
    let root = root();
    let mut r = Vec::new();
    let mut i = 0;
    while i < s.len() {
      let mut t = root;
      let mut c: Option<&JyutpingValue> = None;
      let mut k = i;
      for (j, &ch) in s.iter().enumerate().skip(i) {
        let Some(u) = t.get(ch) else {
          break;
        };
        t = u;
        if let Some(v) = self.get_value(t)
          && !v.is_empty()
        {
          c = Some(&v[0]);
          k = j;
        }
      }
      if k == i {
        r.push((s[i].to_string(), c.cloned()));
        i += 1;
      } else if let Some(c) = c {
        let n = i;
        while i <= k {
          r.push((
            s[i].to_string(),
            Some(JyutpingValue::Jyutping(c.getitem(i - n).clone())),
          ));
          i += 1;
        }
      }
    }
    r
  }

  /// Port of `Trie.get_value`.
  pub fn get_value<'a>(&self, n: &'a Node) -> Option<&'a Vec<JyutpingValue>> {
    n.v.as_ref()
  }
}
