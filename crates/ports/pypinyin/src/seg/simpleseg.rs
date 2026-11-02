//! Port of `pypinyin/seg/simpleseg.py`.

use crate::constants::{PHRASES_DICT, RE_HANS};
use crate::core::Hans;
use crate::seg::mmseg;

/// Port of `pypinyin.seg.simpleseg.seg`: split by Han / non-Han, then split
/// Han runs with [`mmseg::seg`].
pub fn seg(hans: &str) -> Vec<String> {
  let hans = simple_seg(hans);
  let mut ret = Vec::new();
  for x in hans {
    if !RE_HANS.is_match(&x) {
      // 没有拼音的字符，不再参与二次分词
      ret.push(x);
    } else if !PHRASES_DICT.is_empty() {
      ret.extend(mmseg::seg.cut(&x));
    } else {
      // 禁用了词语库，不分词
      ret.push(x);
    }
  }
  ret
}

/// Port of `pypinyin.seg.simpleseg.simple_seg`: 将传入的字符串按是否是汉字来分割.
///
/// Accepts a string or a list of strings, like Python.
pub fn simple_seg<'a>(hans: impl Into<Hans<'a>>) -> Vec<String> {
  match hans.into() {
    Hans::Str(s) => _seg(s),
    Hans::List(list) => {
      if list.len() == 1 {
        return simple_seg(list[0]);
      }
      list.iter().flat_map(|x| simple_seg(*x)).collect()
    }
  }
}

/// Port of `pypinyin.seg.simpleseg._seg`: 按是否是汉字进行分词.
pub fn _seg(chars: &str) -> Vec<String> {
  let mut s = String::new(); // 保存一个词
  let mut ret = Vec::new(); // 分词结果
  let mut flag = 0; // 上一个字符是什么? 0: 汉字, 1: 不是汉字

  let mut buf = [0u8; 4];
  for (n, c) in chars.chars().enumerate() {
    if RE_HANS.is_match(c.encode_utf8(&mut buf)) {
      // 汉字, 确定 flag 的初始值
      if n == 0 {
        // 第一个字符
        flag = 0;
      }

      if flag == 0 {
        s.push(c);
      } else {
        // 上一个字符不是汉字, 分词
        ret.push(std::mem::take(&mut s));
        flag = 0;
        s.push(c);
      }
    } else {
      // 不是汉字
      if n == 0 {
        // 第一个字符, 确定 flag 的初始值
        flag = 1;
      }

      if flag == 1 {
        s.push(c);
      } else {
        // 上一个字符是汉字, 分词
        ret.push(std::mem::take(&mut s));
        flag = 1;
        s.push(c);
      }
    }
  }

  ret.push(s); // 最后的词
  ret
}
