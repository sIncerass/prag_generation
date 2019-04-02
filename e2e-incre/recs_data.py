import importlib
import os
import sys
from os.path import join, exists
import numpy as np
import pickle as pk
from components.utils.config import load_config, fix_seed
from components.utils.log import set_logger
from components.utils.serialization import load_model, make_model_dir
from components.utils.serialization import save_config, save_predictions_txt
from components.constants import NAME_TOKEN, NEAR_TOKEN
from collections import Counter
from components.constants import BOS_ID, EOS_ID, PAD_ID
def pad_snt(snt_ids_trunc, max_len):
    snt_ids_trunc_pad = snt_ids_trunc + [PAD_ID] * (max_len - len(snt_ids_trunc))
    return snt_ids_trunc_pad

MR_FIELDS = ["name", "familyFriendly", "eatType", "food", "priceRange", "near", "area", "customer rating"]
data_params = {
  "train_data": "/data/sheng/e2e/data/trainset.csv", 
   "dev_data": "/data/sheng/e2e/data/devset.csv",
   "test_data": "/data/sheng/e2e/data/testset.csv",
   "max_src_len": 50,
   "max_tgt_len": 50
}

data_module = "components.data.e2e_data_MLP"
DataClass = importlib.import_module(data_module).component

data = DataClass(data_params)
data.setup()
vocab = data.vocab
id2tok, tok2id = vocab.id2tok, vocab.tok2id
pk.dump(id2tok, open("recs/vocab.pkl", "wb"))
dev_data, dev_lexical = np.array(data.dev), np.array(data.lexicalizations['dev'])
train_data, train_lexical = np.array(data.train), np.array(data.lexicalizations['train'])
test_data, test_lexical = np.array(data.test), np.array(data.lexicalizations['test'])
#"""
pk.dump([train_data, train_lexical], open("recs/train_ori.pkl", "wb"))
pk.dump([test_data, test_lexical], open("recs/test_ori.pkl", "wb"))
pk.dump([dev_data, dev_lexical], open("recs/dev_ori.pkl", "wb"))
#"""
print(dev_data.shape, dev_lexical.shape)
print(len(id2tok), id2tok[5], len(dev_data[0]), dev_data[1][0])
def parse_data_recs(mr_value_vocab2id, mr_value_num, data_process):
    global MR_FIELDS, id2tok
    data_new, it_num, fa_num = [[], []], 0, 0
    data_new_x, data_new_y = [], [[] for i in range(len(MR_FIELDS))]
    print(len(data_process[0]), len(data_process[1]))
    for data_src, data_tgt in zip(data_process[0], data_process[1]):
        data_new_tmp = []
        data_tgt = pad_snt(data_tgt, data_params["max_tgt_len"])
        for mr_id, mr_value_vocab in enumerate(data_src):
            #print(mr_id, mr_value_vocab)
            mr_cur_class = np.zeros( mr_value_num[MR_FIELDS[mr_id]] )
            mr_cur_class[ mr_value_vocab2id[ MR_FIELDS[mr_id] ][mr_value_vocab] ] = 1
            #data_new_tmp.append( np.array( mr_cur_class ) )
            data_new_y[ mr_id ].append( mr_value_vocab2id[ MR_FIELDS[mr_id] ][mr_value_vocab] )#np.array( mr_cur_class ) )
        try:
            assert len(data_tgt) == data_params["max_tgt_len"] and (32 in data_tgt) #or 35 in data_tgt)
        except:
            fa_num += 1
            data_new_y[0][ -1 ] = 1
        data_new_x.append( data_tgt )
        it_num += 1
    data_new_y = [np.array(x) for x in data_new_y]
    print("No special Name or Near", fa_num * 100.0 / it_num)
    return [data_new_x, data_new_y]

def tidy_recs():
    global data, MR_FIELDS
    all_input = data.dev[0] + data.train[0] + data.test[0]
    all_input = set([tuple(x) for x in all_input])
    print(len(all_input), len(data.lexicalizations['train']), len(data.lexicalizations['test']), len(data.lexicalizations['dev']))
    lexical = data.lexicalizations['train'] + data.lexicalizations['test'] + data.lexicalizations['dev']
    mr_value = {x:[] for x in MR_FIELDS}
    for _ in all_input:
        for mr_id, mr_f in enumerate(_):
            mr_value[ MR_FIELDS[mr_id] ].append(mr_f)
    
    mr_value_id2vocab, mr_value_vocab2id, mr_value_num = {x:{} for x in MR_FIELDS}, {x:{} for x in MR_FIELDS}, {x:{} for x in MR_FIELDS}
    for mr_f in mr_value:
        c = Counter( mr_value[mr_f] ).most_common()
        print(c)
        mr_value_num[ mr_f ] = len(c)
        for c_idx, (c_v, _) in enumerate(c):
            mr_value_id2vocab[ mr_f ][c_idx] = c_v
            mr_value_vocab2id[ mr_f ][c_v] = c_idx
    pk.dump([mr_value_id2vocab, mr_value_vocab2id, mr_value_num], open("recs/mr_value.pkl", "wb"))
    data_new_train = parse_data_recs(mr_value_vocab2id, mr_value_num, data.train)
    data_new_val = parse_data_recs(mr_value_vocab2id, mr_value_num, data.dev)
    pk.dump([data_new_train, data.lexicalizations['train']], open("recs/train.pkl", "wb"))
    # a more straight forward way is to just change dir of devset to testset, then modift the according filename
    #data_new_test = parse_data_recs(mr_value_vocab2id, mr_value_num, data.test)
    #pk.dump([data_new_test, data.lexicalizations['test']], open("recs/test.pkl", "wb"))
    pk.dump([data_new_val, data.lexicalizations['dev']], open("recs/dev.pkl", "wb"))

def w2v(dim):
    global data
    import gensim
    sentences =  data.dev[1] + data.train[1]
    sentences = [ [str(x) for x in s] for s in sentences ]
    model = gensim.models.Word2Vec(size=dim, min_count=0, workers=16, sg=1)
    model.build_vocab(sentences)
    model.train(sentences,
                    total_examples=model.corpus_count, epochs=model.iter)
    model.wv.save_word2vec_format(join(
            "recs/", 'word2vec.{}d.{}k.w2v'.format(dim, len(model.wv.vocab)//1000)))

def pre_w2v(dim):         
    global id2tok
    w2v_pre = {}
    with open('recs/word2vec.{}d.3k.w2v'.format(dim), encoding="utf8") as stream:
        for idx, line in enumerate(stream):
            if idx == 0:
                continue
            split = line.rstrip().split(" ")
            word = int(split[0])
            vector = np.array([float(num) for num in split[1:]])
            w2v_pre[word] = vector
    p = sorted(w2v_pre.keys())
    embeds = [np.zeros(dim), np.random.uniform(-0.25, 0.25, dim), np.random.uniform(-0.25, 0.25, dim), np.random.uniform(-0.25, 0.25, dim)]
    for idx in range(4, len(id2tok)):
        #print(id2tok[idx])
        embeds.append( w2v_pre.get(idx, np.random.uniform(-0.25, 0.25, dim)) )
    pk.dump(embeds, open("recs/w2v.{}d.pkl".format(dim), "wb"))
# print(id2tok[32], id2tok[35], [id2tok[x] for x in range(10)])
print(data.dev[1][1])
tidy_recs()
w2v(128)
pre_w2v(128)