# -*- encoding: utf-8 -*-
# @Author: Jinghui Qin
# @Time: 2021/10/15
# @File: try_num.py
from transformers import BertTokenizer, AdamW

bert_path = "./pretrained_lm/chinese-bert-wwm"

bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

bert_tokenizer.add_tokens(['[NUM]'])
print(bert_tokenizer.vocab)

print(bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.encode("小 明 在 一天 中 测量 了 [NUM] 次 气温 ， 分别 是 ： [NUM] ° C 、 [NUM] ° C 、 [NUM] ° C 、 [NUM] ° C 、 [NUM] ° C 、 [NUM] ° C ． 这 一天 的 平均气温 = 多少 ° C ．")))
