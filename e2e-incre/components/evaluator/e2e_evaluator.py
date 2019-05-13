import logging
from collections import Counter
from components.constants import NAME_TOKEN, NEAR_TOKEN
from components.data.common import ids2var
import torch

logger = logging.getLogger('experiment')


class BaseEvaluator(object):
    """
    Base class containing methods for evaluation of E2E NLG models.
    """

    def __init__(self, config):
        self.config = config or dict()

    def label2snt(self, id2word, ids):
        tokens = [id2word[t] for t in ids]
        return tokens, ' '.join(tokens)

    def predict_one(self, model, src_snt_ids, beam_size=None):
        input_var = ids2var(src_snt_ids, -1, 1, addEOS=True)  # batch_size = 1; cudified
        #print(input_var)
        #input_var = input_var.cpu()#.type(torch.cuda.LongTensor) 
        #print(input_var)
        if beam_size:
            output_ids, attention_weights = model.predict_beam(input_var, beam_size=beam_size)
        else:   
            output_ids, attention_weights = model.predict(input_var)
        return output_ids, attention_weights
    
    def predict_dis(self, model, src_snt_ids, dis_sns, alpha=1.0):
        input_var = ids2var(src_snt_ids, -1, 1, addEOS=True)
        input_dis = [ids2var(x, -1, 1, addEOS=True) for x in dis_sns]
        #print(input_dis)
        output_ids, attention_weights = model.predict_dis(input_var, input_dis, alpha)
        return output_ids, attention_weights

    def evaluate_model(self, model, dev_data, dev_mr=None, beam_size=None, alpha=1.0, dis=True):
        """
        Evaluating model on multi-ref data
        :param model:
        :param dev_data:
        :return:
        """
        if beam_size is not None:
            decoded_ids, decoded_attn_weights = [[] for _ in range(beam_size)], [[] for _ in range(beam_size)]
            curr_x_ids = dev_mr[0]
            out_ids, scores = self.predict_one(model, dev_data[0], beam_size)
            for _ in range(beam_size):
                decoded_ids[_].append( out_ids[_] )
                decoded_attn_weights[_].append( scores[_] )
            for cur_id, snt_ids in enumerate(dev_data[1:]):
                if dev_mr[cur_id+1] == curr_x_ids:#snt_ids == curr_x_ids:
                    continue
                else:
                    out_ids, scores = self.predict_one(model, snt_ids, beam_size)
                    for _ in range(beam_size):
                        decoded_ids[_].append( out_ids[_] )
                        decoded_attn_weights[_].append( scores[_] )
                    curr_x_ids = dev_mr[cur_id]
        elif dis:
            batch_size = 64 
            dis_x, decoded_ids = [0], []
            decoded_attn_weights = []
            cur_start, cur_tmp, curr_x_ids = 0, dev_data[0], dev_mr[0]
            for cur_id, snt_ids in enumerate(dev_data[1:]):
                if dev_mr[cur_id + 1] == cur_tmp:
                    continue
                else:
                    cur_tmp = dev_mr[cur_id + 1]
                    dis_x.append( cur_id + 1 )
            dis_ids = dis_x[ (cur_start-int(batch_size/2)) : cur_start ] + dis_x[ (cur_start+1):(cur_start+int(batch_size/2))]
            if cur_start in dis_ids:
                dis_ids.remove(cur_start)
            def mask_(snt_ids, dis_ids, dev_data):
                # mask out all MR attributes
                outs = [5 for _ in range(len(snt_ids))] # 5 is the unk value for MR attribute in the dataset
                for cur_f, cur_k in enumerate(snt_ids):
                    if cur_k == 5 and outs[ cur_f ] == 5:
                        pp = []
                        for dis_id in dis_ids:
                            # replace the unk with other values
                            dis_cur = dev_data[dis_id]
                            if dis_cur[cur_f] != 5:
                                pp.append( dis_cur[cur_f] )
                        #print(Counter(pp).most_common(), dis_ids)
                        outs[ cur_f ] = Counter(pp).most_common()[0][0] if len( Counter(pp).most_common()  ) else 5
                    #else:
                        #pp.append( cur_f )
                return outs
            # """
            p = mask_(snt_ids, dis_ids, dev_data)
            dis_sns = [ dev_data[x] for x in dis_ids]
            # then the first will be the masked
            dis_sns = [p] + dis_sns
            #print(dis_sns)
            out_ids, attn_weights = self.predict_dis(model, dev_data[0], dis_sns, alpha)
            decoded_ids.append(out_ids)
            decoded_attn_weights.append(attn_weights[1])
            for cur_id, snt_ids in enumerate(dev_data[1:]):
                real_id = cur_id + 1
                if dev_mr[real_id] == curr_x_ids:
                    continue
                else:
                    cur_start += 1
                    dis_ids = dis_x[ (cur_start-int(batch_size/2)) : cur_start ] + dis_x[ (cur_start+1): (cur_start+int(batch_size/2))]
                    if real_id in dis_ids:
                        dis_ids.remove( real_id )
                    #print(dis_ids)
                    dis_sns = [ dev_data[x] for x in dis_ids]
                    while snt_ids in dis_sns:
                        #print("prev", dis_sns)
                        dis_sns.remove( snt_ids )
                        #print(dis_sns)
                        #exit()
                    out_ids, attn_weights = self.predict_dis(model, snt_ids, dis_sns, alpha)
                    decoded_ids.append(out_ids)
                    decoded_attn_weights.append(attn_weights[1])
                    curr_x_ids = dev_mr[ real_id ]
        else: # original version
            decoded_ids = []
            decoded_attn_weights = []
            # Make a prediction on the first input
            curr_x_ids = dev_data[0]
            #curr_x_ids = dev_mr[0]

            out_ids, attn_weights = self.predict_one(model, dev_data[0])
            decoded_ids.append(out_ids)
            decoded_attn_weights.append(attn_weights[1])

            # Make predictions on the remaining unique (!) inputs
            for cur_id, snt_ids in enumerate(dev_data[1:]):
                real_id = cur_id + 1
                #if dev_mr[real_id] == curr_x_ids:
                if snt_ids == curr_x_ids:
                    continue

                else:
                    out_ids, attn_weights = self.predict_one(model, snt_ids)
                    decoded_ids.append(out_ids)
                    decoded_attn_weights.append(attn_weights[1])
                    #curr_x_ids = dev_mr[real_id]
                    curr_x_ids = snt_ids
            #print(len(decoded_ids))
        return decoded_ids, decoded_attn_weights

    def lexicalize_predictions(self, all_tokids, data_lexicalizations, id2word):
        """
        Given model predictions from a model, convert numerical ids to tokens,
        substituting placeholder items (NEAR and NAME) with the values in "data_lexicalizations",
        which we created during the data preprocessing step.

        :param all_tokids:
        :param data_lexicalizations:
        :param id2word:
        :return:
        """

        all_tokens = []
        for idx, snt_ids in enumerate(all_tokids):

            this_snt_toks = []
            this_snt_lex = data_lexicalizations[idx]
            
            for t in snt_ids[:-1]:  # excluding </s>
                #print(t, t.data.cpu().tolist())
                t = t.data.item()#cpu().tolist()
                tok = id2word[t]

                if tok == NAME_TOKEN:
                    l = this_snt_lex[0]
                    if not l is None:
                        this_snt_toks.append(l)

                elif tok == NEAR_TOKEN:
                    l = this_snt_lex[1]
                    if not l is None:
                        this_snt_toks.append(l)

                else:
                    this_snt_toks.append(tok)

            all_tokens.append(this_snt_toks)

        return all_tokens


component = BaseEvaluator
