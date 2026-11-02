//! Port of `pypinyin/utils.py`.

/// Port of `pypinyin.utils._remove_dup_items`: drop duplicates, keep order.
pub fn _remove_dup_items(lst: &[String], remove_empty: bool) -> Vec<String> {
  let mut new_lst: Vec<String> = Vec::new();
  for item in lst {
    if remove_empty && item.is_empty() {
      continue;
    }
    if !new_lst.contains(item) {
      new_lst.push(item.clone());
    }
  }
  new_lst
}

/// Port of `pypinyin.utils._remove_dup_and_empty`: dedupe every item list,
/// and use `['']` for lists that end up empty.
pub fn _remove_dup_and_empty(lst_list: &[Vec<String>]) -> Vec<Vec<String>> {
  let mut new_lst_list = Vec::with_capacity(lst_list.len());
  for lst in lst_list {
    let lst = _remove_dup_items(lst, true);
    if !lst.is_empty() {
      new_lst_list.push(lst);
    } else {
      new_lst_list.push(vec![String::new()]);
    }
  }
  new_lst_list
}
