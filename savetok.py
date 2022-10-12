from transformers import AutoTokenizer, AutoConfig
# from transformers import XLNetTokenizer

model_name = "bert-base-uncased"
# tokenizer = XLNetTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

config = AutoConfig.from_pretrained(model_name)

tokenizer_saving_path = 'E:\\Documents\\MEGAsync\\code\\SA\\tokenizers\\BERT\\Base\\tokenizer'
tokenizer.save_pretrained(tokenizer_saving_path)
config.save_pretrained(tokenizer_saving_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_saving_path)