# S: Symbol that shows starting of decoding input
# E: Symbol that shows ending of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]
import json
import pickle
f = open("translation2019zh_train.json")
english_sentences = []
chinese_sentences = []
for line in f.readlines():
        data = json.loads(line)
        english_sentences.append(data["english"])
        chinese_sentences.append(data["chinese"])
assert(len(english_sentences) == len(chinese_sentences))
from bpe import Encoder
en_encoder = Encoder(32000, pct_bpe=0.88)
en_encoder.fit(english_sentences)
zh_encoder = Encoder(32000, pct_bpe=0.88)
zh_encoder.fit(chinese_sentences)
# example = "Slowly and not without struggle, America began to listen."
# print(encoder.tokenize(example))
en_segment_sentences = [en_encoder.tokenize(sent) for sent in english_sentences ]
zh_segment_sentences = [zh_encoder.tokenize(sent) for sent in chinese_sentences]
file = open("zh_segment_sentences", "wb")
pickle.dump(zh_segment_sentences, file)
file.close()
file = open("en_segment_sentences", "wb")
pickle.dump(en_segment_sentences, file)
file.close()    



# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)
