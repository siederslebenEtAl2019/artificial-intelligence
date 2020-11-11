# testing sentencepiece 22/7/2020


import sentencepiece as spm

# s = spm.SentencePieceProcessor(model_file='spm.model')

spm.SentencePieceTrainer.train('--model_type=bpe '
                               '--input=blog_test.txt '
                               '--model_prefix=bpe '
                               '--vocab_size=500 '
                               '--normalization_rule_tsv=normalization_rule.tsv')