# ja_userdic

`userdict.csv` is `GPT_SoVITS/text/ja_userdic/userdict.csv` from GPT-SoVITS
commit `04a44eb` (MIT, Copyright (c) 2024 RVC-Boss), unchanged (md5
`d36bd5ffba62f195d22bf4f1a41cd08f`). It is an Open JTalk (MeCab) user
dictionary: 134,201 rows of
`surface,left_id,right_id,cost,pos,pos_group1,pos_group2,pos_group3,ctype,cform,orig,read,pron,acc/mora_size,chain_rule`,
mostly lowercase fullwidth English words with katakana readings.

Upstream compiles it with `pyopenjtalk.mecab_dict_index` into `user.dict` and
loads it with `pyopenjtalk.update_global_jtalk_with_user_dict`. The port
embeds the CSV and builds a jpreprocess user dictionary from it on first use
(`gpt_sovits_text::japanese::pyopenjtalk::mecab_dict_index`); see
`parity/japanese/README.md` for the rows it adapts.
