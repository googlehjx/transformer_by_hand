from model.decode import greedy_decoder
from data.my_data import loader
from data.my_vocab import tgt_vocab, idx2word
from model.my_model import Transformer

# Test
model = Transformer()
enc_inputs, _, _ = next(iter(loader))
# enc_inputs = enc_inputs.cuda()
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])