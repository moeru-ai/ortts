//! Port of `GPT_SoVITS/text/tone_sandhi.py` (GPT-SoVITS @ 04a44eb).
//!
//! The Python file comes from PaddleSpeech and is Apache-2.0 licensed
//! (Copyright (c) 2021 PaddlePaddle Authors). See `NOTICE`.
//!
//! The port follows the Python control flow line by line, including its
//! quirks. Strings are indexed by Unicode code point, as Python `str` is.
//! Where Python raises `IndexError` (for example a final list that is shorter
//! than the word, an empty final string, or an empty `pos`), the Rust method
//! returns [`IndexError`] with the Python message.
//!
//! Python mutates the `finals` list that the caller passes in. The Rust
//! methods take `finals` by value and return the new list, which is the value
//! that the Python methods return.

use std::collections::HashSet;
use std::fmt;

use pypinyin::{LazyPinyinOptions, Style, lazy_pinyin};

/// A segmented sentence: `(word, pos)` pairs, as `jieba_fast.posseg.lcut`
/// returns them and as the `_merge_*` methods pass them on. Python mixes
/// tuples and two-item lists; both are a `(String, String)` here.
pub type Seg = Vec<(String, String)>;

/// The `IndexError` that Python raises inside `ToneSandhi`. Holds the Python
/// message: `"list index out of range"` or `"string index out of range"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexError(pub &'static str);

impl IndexError {
  const LIST: Self = Self("list index out of range");
  const STRING: Self = Self("string index out of range");
}

impl fmt::Display for IndexError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "IndexError: {}", self.0)
  }
}

impl std::error::Error for IndexError {}

/// The `jieba.cut_for_search` call of `_split_word`.
pub type CutForSearch = Box<dyn Fn(&str) -> Vec<String> + Send + Sync>;

/// Port of `ToneSandhi`.
pub struct ToneSandhi {
  /// Port of `self.must_neural_tone_words`.
  pub must_neural_tone_words: HashSet<&'static str>,
  /// Port of `self.must_not_neural_tone_words`.
  pub must_not_neural_tone_words: HashSet<&'static str>,
  /// Port of `self.punc`.
  pub punc: &'static str,
  /// The module-level `jieba.cut_for_search` (`jieba_fast`, `HMM=True`)
  /// that `_split_word` calls. It is a field so that a caller can replace
  /// the segmenter, as Python code can monkeypatch `jieba.cut_for_search`.
  /// [`ToneSandhi::new`] sets it to `jieba_fast::cut_for_search(word, true)`.
  pub cut_for_search: CutForSearch,
}

impl fmt::Debug for ToneSandhi {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ToneSandhi")
      .field("must_neural_tone_words", &self.must_neural_tone_words.len())
      .field(
        "must_not_neural_tone_words",
        &self.must_not_neural_tone_words.len(),
      )
      .field("punc", &self.punc)
      .finish_non_exhaustive()
  }
}

impl Default for ToneSandhi {
  fn default() -> Self {
    Self::new()
  }
}

/// The items of the `self.must_neural_tone_words` set literal, in source order.
pub const MUST_NEURAL_TONE_WORDS: &[&str] = &[
  "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝", "难为",
  "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊", "里头", "部分",
  "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气", "过去", "软和", "转悠", "踏实",
  "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号", "认识", "规矩", "见识", "裁缝",
  "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄",
  "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在", "膏药", "脾气",
  "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉",
  "耳朵", "老爷", "老实", "老婆", "老头", "老太", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火",
  "累赘", "糨糊", "糊涂", "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语",
  "笑话", "窟窿", "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗",
  "砚台", "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算",
  "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨", "琉璃",
  "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快", "爱人", "热闹",
  "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚",
  "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃",
  "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠", "朋友", "月饼", "月亮",
  "暖和", "明白", "时候", "新鲜", "故事", "收拾", "收成", "提防", "挖苦", "挑剔", "指甲", "指头",
  "拾掇", "拳头", "拨弄", "招牌", "招呼", "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打点",
  "打扮", "打听", "打发", "扎实", "扁担", "戒指", "懒得", "意识", "意思", "情形", "悟性", "怪物",
  "思量", "怎么", "念头", "念叨", "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通",
  "应酬", "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫",
  "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气",
  "实在", "官司", "学问", "学生", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘",
  "姐夫", "妯娌", "妥当", "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
  "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔", "嘀咕",
  "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚", "告诉", "告示",
  "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋", "厚道", "厉害", "千斤",
  "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气", "前头", "刺猬", "刺激", "别扭",
  "利落", "利索", "利害", "分析", "出息", "凑合", "凉快", "冷战", "冤枉", "冒失", "养活", "关系",
  "先生", "兄弟", "便宜", "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么",
  "人家", "亲戚", "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西",
  "东家", "世故", "不由", "不在", "下水", "下巴", "上头", "上司", "丈夫", "丈人", "一辈", "那个",
  "菩萨", "父亲", "母亲", "咕噜", "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅",
  "幸福", "熟悉", "计划", "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰",
  "拖沓", "寒碜", "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
];

/// The items of the `self.must_not_neural_tone_words` set literal, in source
/// order.
pub const MUST_NOT_NEURAL_TONE_WORDS: &[&str] = &[
  "男子",
  "女子",
  "分子",
  "原子",
  "量子",
  "莲子",
  "石子",
  "瓜子",
  "电子",
  "人人",
  "虎虎",
  "幺幺",
  "干嘛",
  "学子",
  "哈哈",
  "数数",
  "袅袅",
  "局地",
  "以下",
  "娃哈哈",
  "花花草草",
  "留得",
  "耕地",
  "想想",
  "熙熙",
  "攘攘",
  "卵子",
  "死死",
  "冉冉",
  "恳恳",
  "佼佼",
  "吵吵",
  "打打",
  "考考",
  "整整",
  "莘莘",
  "落地",
  "算子",
  "家家户户",
  "青青",
];

/// The value of `self.punc`.
pub const PUNC: &str = "：，；。？！“”‘’':,;.?!";

// ---------------------------------------------------------------------------
// Python `str` / `list` helpers (code point based).

fn chars(s: &str) -> Vec<char> {
  s.chars().collect()
}

/// `s[-1]` for a `str`.
fn last_char(s: &str) -> Result<char, IndexError> {
  s.chars().next_back().ok_or(IndexError::STRING)
}

/// `s[0]` for a `str`.
fn first_char(s: &str) -> Result<char, IndexError> {
  s.chars().next().ok_or(IndexError::STRING)
}

/// `s[:-1] + tone`.
fn with_tone(s: &str, tone: char) -> String {
  let mut out: String = s.to_string();
  out.pop();
  out.push(tone);
  out
}

/// Python list index (negative indices count from the end).
fn py_index(len: usize, idx: isize) -> Result<usize, IndexError> {
  let i = if idx < 0 { idx + len as isize } else { idx };
  if i < 0 || i as usize >= len {
    Err(IndexError::LIST)
  } else {
    Ok(i as usize)
  }
}

/// `finals[idx] = finals[idx][:-1] + tone`.
fn set_tone(finals: &mut [String], idx: isize, tone: char) -> Result<(), IndexError> {
  let i = py_index(finals.len(), idx)?;
  finals[i] = with_tone(&finals[i], tone);
  Ok(())
}

/// `finals[idx]`.
fn item(finals: &[String], idx: isize) -> Result<&str, IndexError> {
  let i = py_index(finals.len(), idx)?;
  Ok(&finals[i])
}

/// `word[-2:]` for a `str`.
fn last_two(word: &[char]) -> String {
  word[word.len().saturating_sub(2)..].iter().collect()
}

/// `[finals[:n], finals[n:]]`.
fn split_at(finals: &[String], n: usize) -> [Vec<String>; 2] {
  let n = n.min(finals.len());
  [finals[..n].to_vec(), finals[n..].to_vec()]
}

/// `lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)`.
fn finals_tone3(word: &str) -> Vec<String> {
  lazy_pinyin(
    word,
    &LazyPinyinOptions {
      style: Style::FINALS_TONE3,
      neutral_tone_with_five: true,
      ..LazyPinyinOptions::default()
    },
  )
}

fn in_ranges(ch: char, ranges: &[(char, char)]) -> bool {
  ranges
    .binary_search_by(|&(lo, hi)| {
      if hi < ch {
        std::cmp::Ordering::Less
      } else if lo > ch {
        std::cmp::Ordering::Greater
      } else {
        std::cmp::Ordering::Equal
      }
    })
    .is_ok()
}

/// Python `str.isnumeric()` for one character (Unicode 14.0.0 in Python 3.11).
pub fn py_isnumeric(ch: char) -> bool {
  in_ranges(ch, PY_NUMERIC_RANGES)
}

impl ToneSandhi {
  /// Port of `ToneSandhi.__init__`.
  pub fn new() -> Self {
    Self {
      must_neural_tone_words: MUST_NEURAL_TONE_WORDS.iter().copied().collect(),
      must_not_neural_tone_words: MUST_NOT_NEURAL_TONE_WORDS.iter().copied().collect(),
      punc: PUNC,
      cut_for_search: Box::new(|word| jieba_fast::cut_for_search(word, true)),
    }
  }

  fn is_must_neural(&self, word: &[char]) -> bool {
    let w: String = word.iter().collect();
    self.must_neural_tone_words.contains(w.as_str())
      || self
        .must_neural_tone_words
        .contains(last_two(word).as_str())
  }

  /// Port of `ToneSandhi._neural_sandhi(word, pos, finals)`.
  pub fn _neural_sandhi(
    &self,
    word: &str,
    pos: &str,
    mut finals: Vec<String>,
  ) -> Result<Vec<String>, IndexError> {
    let w = chars(word);
    // reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for (j, &item_) in w.iter().enumerate() {
      if j >= 1
        && item_ == w[j - 1]
        && matches!(first_char(pos)?, 'n' | 'v' | 'a')
        && !self.must_not_neural_tone_words.contains(word)
      {
        set_tone(&mut finals, j as isize, '5')?;
      }
    }
    let ge_idx: isize = w.iter().position(|&c| c == '个').map_or(-1, |i| i as isize);
    let last = w.last().copied();
    if !w.is_empty() && "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶".contains(last.unwrap())
    {
      set_tone(&mut finals, -1, '5')?;
    } else if !w.is_empty() && "的地得".contains(last.unwrap()) {
      set_tone(&mut finals, -1, '5')?;
    // e.g. 走了, 看着, 去过
    } else if w.len() == 1 && "了着过".contains(word) && matches!(pos, "ul" | "uz" | "ug") {
      set_tone(&mut finals, -1, '5')?;
    } else if w.len() > 1
      && "们子".contains(last.unwrap())
      && matches!(pos, "r" | "n")
      && !self.must_not_neural_tone_words.contains(word)
    {
      set_tone(&mut finals, -1, '5')?;
    // e.g. 桌上, 地下, 家里
    } else if w.len() > 1 && "上下里".contains(last.unwrap()) && matches!(pos, "s" | "l" | "f") {
      set_tone(&mut finals, -1, '5')?;
    // e.g. 上来, 下去
    } else if w.len() > 1
      && "来去".contains(last.unwrap())
      && "上下进出回过起开".contains(w[w.len() - 2])
    {
      set_tone(&mut finals, -1, '5')?;
    // 个做量词
    } else if (ge_idx >= 1 && {
      let prev = w[ge_idx as usize - 1];
      py_isnumeric(prev) || "几有两半多各整每做是".contains(prev)
    }) || word == "个"
    {
      set_tone(&mut finals, ge_idx, '5')?;
    } else if self.is_must_neural(&w) {
      set_tone(&mut finals, -1, '5')?;
    }

    let word_list = self._split_word(word)?;
    let n0 = word_list[0].chars().count();
    let mut finals_list = split_at(&finals, n0);
    for (i, word) in word_list.iter().enumerate() {
      // conventional neural in Chinese
      if self.is_must_neural(&chars(word)) {
        set_tone(&mut finals_list[i], -1, '5')?;
      }
    }
    let [a, b] = finals_list;
    Ok([a, b].concat())
  }

  /// Port of `ToneSandhi._bu_sandhi(word, finals)`.
  pub fn _bu_sandhi(&self, word: &str, mut finals: Vec<String>) -> Result<Vec<String>, IndexError> {
    let w = chars(word);
    // e.g. 看不懂
    if w.len() == 3 && w[1] == '不' {
      set_tone(&mut finals, 1, '5')?;
    } else {
      for (i, &ch) in w.iter().enumerate() {
        // "不" before tone4 should be bu2, e.g. 不怕
        if ch == '不' && i + 1 < w.len() && last_char(item(&finals, i as isize + 1)?)? == '4' {
          set_tone(&mut finals, i as isize, '2')?;
        }
      }
    }
    Ok(finals)
  }

  /// Port of `ToneSandhi._yi_sandhi(word, finals)`.
  pub fn _yi_sandhi(&self, word: &str, mut finals: Vec<String>) -> Result<Vec<String>, IndexError> {
    let w = chars(word);
    // "一" in number sequences, e.g. 一零零, 二一零
    if w.contains(&'一') && w.iter().filter(|&&c| c != '一').all(|&c| py_isnumeric(c)) {
      return Ok(finals);
    // "一" between reduplication words shold be yi5, e.g. 看一看
    } else if w.len() == 3 && w[1] == '一' && w[0] == w[2] {
      set_tone(&mut finals, 1, '5')?;
    // when "一" is ordinal word, it should be yi1
    } else if word.starts_with("第一") {
      set_tone(&mut finals, 1, '1')?;
    } else {
      for (i, &ch) in w.iter().enumerate() {
        if ch == '一' && i + 1 < w.len() {
          // "一" before tone4 should be yi2, e.g. 一段
          if last_char(item(&finals, i as isize + 1)?)? == '4' {
            set_tone(&mut finals, i as isize, '2')?;
          // "一" before non-tone4 should be yi4, e.g. 一天
          } else {
            // "一" 后面如果是标点，还读一声
            if !self.punc.contains(w[i + 1]) {
              set_tone(&mut finals, i as isize, '4')?;
            }
          }
        }
      }
    }
    Ok(finals)
  }

  /// Port of `ToneSandhi._split_word(word)`. Always returns two items.
  pub fn _split_word(&self, word: &str) -> Result<Vec<String>, IndexError> {
    let mut word_list = (self.cut_for_search)(word);
    // sorted(..., key=len): a stable sort by code point count
    word_list.sort_by_key(|w| w.chars().count());
    let first_subword = word_list.first().ok_or(IndexError::LIST)?.clone();
    let w = chars(word);
    let n = first_subword.chars().count();
    let new_word_list = if word.find(first_subword.as_str()) == Some(0) {
      let second_subword: String = w[n.min(w.len())..].iter().collect();
      vec![first_subword, second_subword]
    } else {
      // word[:-len(first_subword)]; `word[:-0]` is `word[:0]`
      let end = if n == 0 { 0 } else { w.len().saturating_sub(n) };
      let second_subword: String = w[..end].iter().collect();
      vec![second_subword, first_subword]
    };
    Ok(new_word_list)
  }

  /// Port of `ToneSandhi._three_sandhi(word, finals)`.
  pub fn _three_sandhi(
    &self,
    word: &str,
    mut finals: Vec<String>,
  ) -> Result<Vec<String>, IndexError> {
    let len = word.chars().count();
    if len == 2 && self._all_tone_three(&finals)? {
      set_tone(&mut finals, 0, '2')?;
    } else if len == 3 {
      let word_list = self._split_word(word)?;
      let n0 = word_list[0].chars().count();
      if self._all_tone_three(&finals)? {
        //  disyllabic + monosyllabic, e.g. 蒙古/包
        if n0 == 2 {
          set_tone(&mut finals, 0, '2')?;
          set_tone(&mut finals, 1, '2')?;
        //  monosyllabic + disyllabic, e.g. 纸/老虎
        } else if n0 == 1 {
          set_tone(&mut finals, 1, '2')?;
        }
      } else {
        let mut finals_list = split_at(&finals, n0);
        // `if len(finals_list) == 2` is always true
        for i in 0..2 {
          let sub = &finals_list[i];
          // e.g. 所有/人
          if self._all_tone_three(sub)? && sub.len() == 2 {
            set_tone(&mut finals_list[i], 0, '2')?;
          // e.g. 好/喜欢
          } else if i == 1
            && !self._all_tone_three(sub)?
            && last_char(item(sub, 0)?)? == '3'
            && last_char(item(&finals_list[0], -1)?)? == '3'
          {
            set_tone(&mut finals_list[0], -1, '2')?;
          }
          finals = finals_list.concat();
        }
      }
    // split idiom into two words who's length is 2
    } else if len == 4 {
      let finals_list = split_at(&finals, 2);
      finals = Vec::new();
      for mut sub in finals_list {
        if self._all_tone_three(&sub)? {
          set_tone(&mut sub, 0, '2')?;
        }
        finals.extend(sub);
      }
    }
    Ok(finals)
  }

  /// Port of `ToneSandhi._all_tone_three(finals)`. `all()` stops at the
  /// first final that is not tone 3, so a later empty final does not raise.
  pub fn _all_tone_three<S: AsRef<str>>(&self, finals: &[S]) -> Result<bool, IndexError> {
    for x in finals {
      if last_char(x.as_ref())? != '3' {
        return Ok(false);
      }
    }
    Ok(true)
  }

  /// Port of `ToneSandhi._merge_bu(seg)`: merge "不" and the word behind it.
  pub fn _merge_bu(&self, seg: &[(String, String)]) -> Seg {
    let mut new_seg = Vec::new();
    let mut last_word = String::new();
    for (word, pos) in seg {
      let mut word = word.clone();
      if last_word == "不" {
        word = format!("{last_word}{word}");
      }
      if word != "不" {
        new_seg.push((word.clone(), pos.clone()));
      }
      last_word = word;
    }
    if last_word == "不" {
      new_seg.push((last_word, "d".to_string()));
    }
    new_seg
  }

  /// Port of `ToneSandhi._merge_yi(seg)`.
  ///
  /// Function 1 merges "一" and the reduplication words left and right of it
  /// (`听`, `一`, `听` -> `听一听`); function 2 merges a single "一" and the
  /// word behind it. The `IndexError` branch (`new_seg[-1] = ...` on an empty
  /// `new_seg`) cannot be reached, but is kept as in Python.
  pub fn _merge_yi(&self, seg: &[(String, String)]) -> Result<Seg, IndexError> {
    let mut new_seg: Seg = Vec::new();
    let mut i = 0;
    // function 1
    while i < seg.len() {
      let (word, pos) = &seg[i];
      let mut merged = false;
      if i >= 1 && word == "一" && i + 1 < seg.len() {
        let last = new_seg.last().unwrap_or(&seg[i - 1]).clone();
        if last.0 == seg[i + 1].0 && last.1 == "v" && seg[i + 1].1 == "v" {
          let combined = format!("{}一{}", last.0, seg[i + 1].0);
          *new_seg.last_mut().ok_or(IndexError::LIST)? = (combined, last.1);
          i += 2;
          merged = true;
        }
      }
      if !merged {
        new_seg.push((word.clone(), pos.clone()));
        i += 1;
      }
    }
    let seg = new_seg;
    let mut new_seg: Seg = Vec::new();
    // function 2
    for (word, pos) in seg {
      match new_seg.last_mut() {
        Some(last) if last.0 == "一" => last.0.push_str(&word),
        _ => new_seg.push((word, pos)),
      }
    }
    Ok(new_seg)
  }

  /// Port of `ToneSandhi._merge_continuous_three_tones(seg)`: merge two
  /// neighbouring words that are both all tone three.
  pub fn _merge_continuous_three_tones(&self, seg: &[(String, String)]) -> Result<Seg, IndexError> {
    let mut new_seg: Seg = Vec::new();
    let sub_finals_list: Vec<Vec<String>> =
      seg.iter().map(|(word, _)| finals_tone3(word)).collect();
    let mut merge_last = vec![false; seg.len()];
    for (i, (word, pos)) in seg.iter().enumerate() {
      if i >= 1
        && self._all_tone_three(&sub_finals_list[i - 1])?
        && self._all_tone_three(&sub_finals_list[i])?
        && !merge_last[i - 1]
      {
        // if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
        if !self._is_reduplication(&seg[i - 1].0)
          && seg[i - 1].0.chars().count() + seg[i].0.chars().count() <= 3
        {
          new_seg
            .last_mut()
            .ok_or(IndexError::LIST)?
            .0
            .push_str(&seg[i].0);
          merge_last[i] = true;
        } else {
          new_seg.push((word.clone(), pos.clone()));
        }
      } else {
        new_seg.push((word.clone(), pos.clone()));
      }
    }
    Ok(new_seg)
  }

  /// Port of `ToneSandhi._is_reduplication(word)`.
  pub fn _is_reduplication(&self, word: &str) -> bool {
    let w = chars(word);
    w.len() == 2 && w[0] == w[1]
  }

  /// Port of `ToneSandhi._merge_continuous_three_tones_2(seg)`: merge two
  /// neighbouring words when the last syllable of the first and the first
  /// syllable of the second are tone three.
  pub fn _merge_continuous_three_tones_2(
    &self,
    seg: &[(String, String)],
  ) -> Result<Seg, IndexError> {
    let mut new_seg: Seg = Vec::new();
    let sub_finals_list: Vec<Vec<String>> =
      seg.iter().map(|(word, _)| finals_tone3(word)).collect();
    let mut merge_last = vec![false; seg.len()];
    for (i, (word, pos)) in seg.iter().enumerate() {
      if i >= 1
        && last_char(item(&sub_finals_list[i - 1], -1)?)? == '3'
        && last_char(item(&sub_finals_list[i], 0)?)? == '3'
        && !merge_last[i - 1]
      {
        // if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
        if !self._is_reduplication(&seg[i - 1].0)
          && seg[i - 1].0.chars().count() + seg[i].0.chars().count() <= 3
        {
          new_seg
            .last_mut()
            .ok_or(IndexError::LIST)?
            .0
            .push_str(&seg[i].0);
          merge_last[i] = true;
        } else {
          new_seg.push((word.clone(), pos.clone()));
        }
      } else {
        new_seg.push((word.clone(), pos.clone()));
      }
    }
    Ok(new_seg)
  }

  /// Port of `ToneSandhi._merge_er(seg)`.
  pub fn _merge_er(&self, seg: &[(String, String)]) -> Seg {
    let mut new_seg: Seg = Vec::new();
    for (i, (word, pos)) in seg.iter().enumerate() {
      if i >= 1 && word == "儿" && seg[i - 1].0 != "#" {
        // `new_seg` is never empty here: every earlier item appended or merged
        new_seg.last_mut().unwrap().0.push_str(&seg[i].0);
      } else {
        new_seg.push((word.clone(), pos.clone()));
      }
    }
    new_seg
  }

  /// Port of `ToneSandhi._merge_reduplication(seg)`.
  pub fn _merge_reduplication(&self, seg: &[(String, String)]) -> Seg {
    let mut new_seg: Seg = Vec::new();
    for (word, pos) in seg {
      match new_seg.last_mut() {
        Some(last) if *word == last.0 => last.0.push_str(word),
        _ => new_seg.push((word.clone(), pos.clone())),
      }
    }
    new_seg
  }

  /// Port of `ToneSandhi.pre_merge_for_modify(seg)`.
  ///
  /// As in Python, a failing `_merge_yi`, `_merge_continuous_three_tones` or
  /// `_merge_continuous_three_tones_2` prints `"<name> failed"` to stdout and
  /// leaves `seg` as it was before that step.
  pub fn pre_merge_for_modify(&self, seg: &[(String, String)]) -> Seg {
    let mut seg = self._merge_bu(seg);
    match self._merge_yi(&seg) {
      Ok(s) => seg = s,
      Err(_) => println!("_merge_yi failed"),
    }
    seg = self._merge_reduplication(&seg);
    match self._merge_continuous_three_tones(&seg) {
      Ok(s) => seg = s,
      Err(_) => println!("_merge_continuous_three_tones failed"),
    }
    match self._merge_continuous_three_tones_2(&seg) {
      Ok(s) => seg = s,
      Err(_) => println!("_merge_continuous_three_tones_2 failed"),
    }
    self._merge_er(&seg)
  }

  /// Port of `ToneSandhi.modified_tone(word, pos, finals)`.
  pub fn modified_tone(
    &self,
    word: &str,
    pos: &str,
    finals: Vec<String>,
  ) -> Result<Vec<String>, IndexError> {
    let finals = self._bu_sandhi(word, finals)?;
    let finals = self._yi_sandhi(word, finals)?;
    let finals = self._neural_sandhi(word, pos, finals)?;
    self._three_sandhi(word, finals)
  }
}

// BEGIN GENERATED (gen_isnumeric_table.py, Unicode 14.0.0)
/// Code points where Python `str.isnumeric()` is true.
const PY_NUMERIC_RANGES: &[(char, char)] = &[
  ('\u{0030}', '\u{0039}'),
  ('\u{00b2}', '\u{00b3}'),
  ('\u{00b9}', '\u{00b9}'),
  ('\u{00bc}', '\u{00be}'),
  ('\u{0660}', '\u{0669}'),
  ('\u{06f0}', '\u{06f9}'),
  ('\u{07c0}', '\u{07c9}'),
  ('\u{0966}', '\u{096f}'),
  ('\u{09e6}', '\u{09ef}'),
  ('\u{09f4}', '\u{09f9}'),
  ('\u{0a66}', '\u{0a6f}'),
  ('\u{0ae6}', '\u{0aef}'),
  ('\u{0b66}', '\u{0b6f}'),
  ('\u{0b72}', '\u{0b77}'),
  ('\u{0be6}', '\u{0bf2}'),
  ('\u{0c66}', '\u{0c6f}'),
  ('\u{0c78}', '\u{0c7e}'),
  ('\u{0ce6}', '\u{0cef}'),
  ('\u{0d58}', '\u{0d5e}'),
  ('\u{0d66}', '\u{0d78}'),
  ('\u{0de6}', '\u{0def}'),
  ('\u{0e50}', '\u{0e59}'),
  ('\u{0ed0}', '\u{0ed9}'),
  ('\u{0f20}', '\u{0f33}'),
  ('\u{1040}', '\u{1049}'),
  ('\u{1090}', '\u{1099}'),
  ('\u{1369}', '\u{137c}'),
  ('\u{16ee}', '\u{16f0}'),
  ('\u{17e0}', '\u{17e9}'),
  ('\u{17f0}', '\u{17f9}'),
  ('\u{1810}', '\u{1819}'),
  ('\u{1946}', '\u{194f}'),
  ('\u{19d0}', '\u{19da}'),
  ('\u{1a80}', '\u{1a89}'),
  ('\u{1a90}', '\u{1a99}'),
  ('\u{1b50}', '\u{1b59}'),
  ('\u{1bb0}', '\u{1bb9}'),
  ('\u{1c40}', '\u{1c49}'),
  ('\u{1c50}', '\u{1c59}'),
  ('\u{2070}', '\u{2070}'),
  ('\u{2074}', '\u{2079}'),
  ('\u{2080}', '\u{2089}'),
  ('\u{2150}', '\u{2182}'),
  ('\u{2185}', '\u{2189}'),
  ('\u{2460}', '\u{249b}'),
  ('\u{24ea}', '\u{24ff}'),
  ('\u{2776}', '\u{2793}'),
  ('\u{2cfd}', '\u{2cfd}'),
  ('\u{3007}', '\u{3007}'),
  ('\u{3021}', '\u{3029}'),
  ('\u{3038}', '\u{303a}'),
  ('\u{3192}', '\u{3195}'),
  ('\u{3220}', '\u{3229}'),
  ('\u{3248}', '\u{324f}'),
  ('\u{3251}', '\u{325f}'),
  ('\u{3280}', '\u{3289}'),
  ('\u{32b1}', '\u{32bf}'),
  ('\u{3405}', '\u{3405}'),
  ('\u{3483}', '\u{3483}'),
  ('\u{382a}', '\u{382a}'),
  ('\u{3b4d}', '\u{3b4d}'),
  ('\u{4e00}', '\u{4e00}'),
  ('\u{4e03}', '\u{4e03}'),
  ('\u{4e07}', '\u{4e07}'),
  ('\u{4e09}', '\u{4e09}'),
  ('\u{4e5d}', '\u{4e5d}'),
  ('\u{4e8c}', '\u{4e8c}'),
  ('\u{4e94}', '\u{4e94}'),
  ('\u{4e96}', '\u{4e96}'),
  ('\u{4ebf}', '\u{4ec0}'),
  ('\u{4edf}', '\u{4edf}'),
  ('\u{4ee8}', '\u{4ee8}'),
  ('\u{4f0d}', '\u{4f0d}'),
  ('\u{4f70}', '\u{4f70}'),
  ('\u{5104}', '\u{5104}'),
  ('\u{5146}', '\u{5146}'),
  ('\u{5169}', '\u{5169}'),
  ('\u{516b}', '\u{516b}'),
  ('\u{516d}', '\u{516d}'),
  ('\u{5341}', '\u{5341}'),
  ('\u{5343}', '\u{5345}'),
  ('\u{534c}', '\u{534c}'),
  ('\u{53c1}', '\u{53c4}'),
  ('\u{56db}', '\u{56db}'),
  ('\u{58f1}', '\u{58f1}'),
  ('\u{58f9}', '\u{58f9}'),
  ('\u{5e7a}', '\u{5e7a}'),
  ('\u{5efe}', '\u{5eff}'),
  ('\u{5f0c}', '\u{5f0e}'),
  ('\u{5f10}', '\u{5f10}'),
  ('\u{62fe}', '\u{62fe}'),
  ('\u{634c}', '\u{634c}'),
  ('\u{67d2}', '\u{67d2}'),
  ('\u{6f06}', '\u{6f06}'),
  ('\u{7396}', '\u{7396}'),
  ('\u{767e}', '\u{767e}'),
  ('\u{8086}', '\u{8086}'),
  ('\u{842c}', '\u{842c}'),
  ('\u{8cae}', '\u{8cae}'),
  ('\u{8cb3}', '\u{8cb3}'),
  ('\u{8d30}', '\u{8d30}'),
  ('\u{9621}', '\u{9621}'),
  ('\u{9646}', '\u{9646}'),
  ('\u{964c}', '\u{964c}'),
  ('\u{9678}', '\u{9678}'),
  ('\u{96f6}', '\u{96f6}'),
  ('\u{a620}', '\u{a629}'),
  ('\u{a6e6}', '\u{a6ef}'),
  ('\u{a830}', '\u{a835}'),
  ('\u{a8d0}', '\u{a8d9}'),
  ('\u{a900}', '\u{a909}'),
  ('\u{a9d0}', '\u{a9d9}'),
  ('\u{a9f0}', '\u{a9f9}'),
  ('\u{aa50}', '\u{aa59}'),
  ('\u{abf0}', '\u{abf9}'),
  ('\u{f96b}', '\u{f96b}'),
  ('\u{f973}', '\u{f973}'),
  ('\u{f978}', '\u{f978}'),
  ('\u{f9b2}', '\u{f9b2}'),
  ('\u{f9d1}', '\u{f9d1}'),
  ('\u{f9d3}', '\u{f9d3}'),
  ('\u{f9fd}', '\u{f9fd}'),
  ('\u{ff10}', '\u{ff19}'),
  ('\u{10107}', '\u{10133}'),
  ('\u{10140}', '\u{10178}'),
  ('\u{1018a}', '\u{1018b}'),
  ('\u{102e1}', '\u{102fb}'),
  ('\u{10320}', '\u{10323}'),
  ('\u{10341}', '\u{10341}'),
  ('\u{1034a}', '\u{1034a}'),
  ('\u{103d1}', '\u{103d5}'),
  ('\u{104a0}', '\u{104a9}'),
  ('\u{10858}', '\u{1085f}'),
  ('\u{10879}', '\u{1087f}'),
  ('\u{108a7}', '\u{108af}'),
  ('\u{108fb}', '\u{108ff}'),
  ('\u{10916}', '\u{1091b}'),
  ('\u{109bc}', '\u{109bd}'),
  ('\u{109c0}', '\u{109cf}'),
  ('\u{109d2}', '\u{109ff}'),
  ('\u{10a40}', '\u{10a48}'),
  ('\u{10a7d}', '\u{10a7e}'),
  ('\u{10a9d}', '\u{10a9f}'),
  ('\u{10aeb}', '\u{10aef}'),
  ('\u{10b58}', '\u{10b5f}'),
  ('\u{10b78}', '\u{10b7f}'),
  ('\u{10ba9}', '\u{10baf}'),
  ('\u{10cfa}', '\u{10cff}'),
  ('\u{10d30}', '\u{10d39}'),
  ('\u{10e60}', '\u{10e7e}'),
  ('\u{10f1d}', '\u{10f26}'),
  ('\u{10f51}', '\u{10f54}'),
  ('\u{10fc5}', '\u{10fcb}'),
  ('\u{11052}', '\u{1106f}'),
  ('\u{110f0}', '\u{110f9}'),
  ('\u{11136}', '\u{1113f}'),
  ('\u{111d0}', '\u{111d9}'),
  ('\u{111e1}', '\u{111f4}'),
  ('\u{112f0}', '\u{112f9}'),
  ('\u{11450}', '\u{11459}'),
  ('\u{114d0}', '\u{114d9}'),
  ('\u{11650}', '\u{11659}'),
  ('\u{116c0}', '\u{116c9}'),
  ('\u{11730}', '\u{1173b}'),
  ('\u{118e0}', '\u{118f2}'),
  ('\u{11950}', '\u{11959}'),
  ('\u{11c50}', '\u{11c6c}'),
  ('\u{11d50}', '\u{11d59}'),
  ('\u{11da0}', '\u{11da9}'),
  ('\u{11fc0}', '\u{11fd4}'),
  ('\u{12400}', '\u{1246e}'),
  ('\u{16a60}', '\u{16a69}'),
  ('\u{16ac0}', '\u{16ac9}'),
  ('\u{16b50}', '\u{16b59}'),
  ('\u{16b5b}', '\u{16b61}'),
  ('\u{16e80}', '\u{16e96}'),
  ('\u{1d2e0}', '\u{1d2f3}'),
  ('\u{1d360}', '\u{1d378}'),
  ('\u{1d7ce}', '\u{1d7ff}'),
  ('\u{1e140}', '\u{1e149}'),
  ('\u{1e2f0}', '\u{1e2f9}'),
  ('\u{1e8c7}', '\u{1e8cf}'),
  ('\u{1e950}', '\u{1e959}'),
  ('\u{1ec71}', '\u{1ecab}'),
  ('\u{1ecad}', '\u{1ecaf}'),
  ('\u{1ecb1}', '\u{1ecb4}'),
  ('\u{1ed01}', '\u{1ed2d}'),
  ('\u{1ed2f}', '\u{1ed3d}'),
  ('\u{1f100}', '\u{1f10c}'),
  ('\u{1fbf0}', '\u{1fbf9}'),
  ('\u{20001}', '\u{20001}'),
  ('\u{20064}', '\u{20064}'),
  ('\u{200e2}', '\u{200e2}'),
  ('\u{20121}', '\u{20121}'),
  ('\u{2092a}', '\u{2092a}'),
  ('\u{20983}', '\u{20983}'),
  ('\u{2098c}', '\u{2098c}'),
  ('\u{2099c}', '\u{2099c}'),
  ('\u{20aea}', '\u{20aea}'),
  ('\u{20afd}', '\u{20afd}'),
  ('\u{20b19}', '\u{20b19}'),
  ('\u{22390}', '\u{22390}'),
  ('\u{22998}', '\u{22998}'),
  ('\u{23b1b}', '\u{23b1b}'),
  ('\u{2626d}', '\u{2626d}'),
  ('\u{2f890}', '\u{2f890}'),
];
// END GENERATED
