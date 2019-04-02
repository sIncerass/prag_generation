# Based on: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import random
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from components.data.common import cuda_if_gpu
from components.constants import BOS_ID, EOS_ID
from components.model import E2ESeq2SeqModel
from components.model.modules.encoders.enc_mlp import EncoderMLP
from components.model.modules.decoders.dec_attn import DecoderRNNAttnBahd


class E2EMLPModel(E2ESeq2SeqModel):
    def set_encoder(self):
        encoder_params = self.config["encoder_params"]
        self.encoder = EncoderMLP(encoder_params)

    def set_decoder(self):
        decoder_rnn_params = self.config["decoder_params"]
        self.decoder = DecoderRNNAttnBahd(rnn_config=decoder_rnn_params,
                                          output_size=self.tgt_vocab_size,
                                          prev_y_dim=self.embedding_dim,
                                          enc_dim=self.encoder.hidden_size,
                                          enc_num_directions=1)

    def forward(self, datum):
        """
        Run the model on one data instance.
        :param datum:
        :return:
        """

        # batch_x_var: SL x B
        # batch_y_var: TL x B
        batch_x_var, batch_y_var = datum

        # Embedding lookup
        encoder_input_embedded = self.embedding_lookup(batch_x_var)  # SL x B x E
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded)

        # Encode embedded input
        # shapes: SL x B x H; 1 x B x H
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        # Decoding using one of two policies:
        #   - use gold standard label as output of the previous step (teacher forcing)
        #   - use previous prediction as output of the previous step (dynamic decoding)
        # See: http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf
        # and the official PyTorch tutorial: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        use_teacher_force = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_force:
            logits = self.decode_teacher(batch_y_var, encoder_hidden, encoder_outputs)
        else:
            logits = self.decode_dynamic(batch_y_var, encoder_hidden, encoder_outputs)

        return logits

    def decode_teacher(self, dec_input_var, encoder_hidden, encoder_outputs):
        """
        Decoding policy 1: feeding the ground truth label as a target
        :param dec_input_var: ground truth labels
        :param encoder_hidden: the last hidden state of the Encoder RNN; (num_layers * num_directions) x B x enc_dim
        :param encoder_outputs: SL x B x enc_dim
        :return:
        """

        dec_len = dec_input_var.size()[0]
        batch_size = dec_input_var.size()[1]
        dec_hidden = encoder_hidden
        dec_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
        predicted_logits = cuda_if_gpu(Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)))

        # Teacher forcing: feed the target as the next input
        for di in range(dec_len):
            prev_y = self.embedding_mat(dec_input)  # embedding lookup of a vector of length = B; result: B x E
            # prev_y = self.dropout(prev_y) # apply dropout
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            # shapes:
            # - dec_output: B, TV
            # - dec_hidden: 1, B, dec_dim
            # - attn_weights: B x SL

            predicted_logits[di] = dec_output  # store this step's outputs
            dec_input = dec_input_var[di]  # next input

        return predicted_logits

    def decode_dynamic(self, dec_input_var, encoder_hidden, encoder_outputs):

        """
        Decoding policy 2: feeding the previous prediction as a target
        :param dec_input_var: ground truth labels (used only to get the shape, not for decoding)
        :param encoder_hidden: the last hidden state of the Encoder RNN; (num_layers * num_directions) x B x enc_dim
        :param encoder_outputs: SL x B x enc_dim
        :return:
        """
        dec_len = dec_input_var.size()[0]
        batch_size = dec_input_var.size()[1]
        dec_hidden = encoder_hidden
        dec_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
        predicted_logits = cuda_if_gpu(Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)))

        # Dynamic decoding: feed the previous prediction as the next input
        for di in range(dec_len):
            prev_y = self.embedding_mat(dec_input)  # B x E
            # prev_y = self.dropout(prev_y)
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            # shapes:
            # - dec_output: B, TV
            # - dec_hidden: 1, B, dec_dim
            # - attn_weights: B x SL

            predicted_logits[di] = dec_output
            topval, topidx = dec_output.data.topk(1)
            dec_input = cuda_if_gpu(Variable(
                torch.LongTensor(topidx.squeeze().cpu().numpy())
            )
            )

        return predicted_logits

    def predict_dis(self, input_var, input_dis, alpha=1.0):
        encoder_input_embedded = self.embedding_lookup(input_var)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        dec_hidden = encoder_hidden[:1]
        # use the prepared first distractor input as the distractor input
        choose_idx, dis_embed = 0, self.embedding_lookup(input_dis[0])
        dis_o, dis_h = self.encoder(dis_embed)
        dis_dec_h, dis_enc_o = dis_h[:1], dis_o
        cur_simi = sum(int(x==y) for x, y in zip(input_var, input_dis[0]))
        """ # choose the most similar of different distractor from the prepared inputs (not necessary)
        for cur_id, cur_dis in enumerate(input_dis[1:]):
            dis_embed = self.embedding_lookup( cur_dis )
            dis_o, dis_h = self.encoder(dis_embed)
            cur_dec_h = dis_h[:1]
            tmp_simi  = sum(int(x==y) for x, y in zip(input_var, cur_dis))  
            #tmp_simi = torch.dot( dec_hidden.view(-1), cur_dec_h.view(-1) ).item()
            #if tmp_simi > cur_simi:
            #    dis_dec_h, dis_enc_o = cur_dec_h, dis_o
            #    choose_idx = cur_id + 1
            #    cur_simi = tmp_simi
            if tmp_simi > cur_simi:
            #if tmp_simi < cur_simi:
               dis_dec_h, dis_enc_o = cur_dec_h, dis_o
               choose_idx = cur_id + 1
               cur_simi = tmp_simi
        #"""
        real_simi = torch.dot( dec_hidden.view(-1), dec_hidden.view(-1) ).item()
        cur_m = cuda_if_gpu( torch.FloatTensor( [ real_simi / (real_simi + cur_simi), cur_simi / (real_simi + cur_simi)] ) )
        cur_m = cur_m.unsqueeze(0)
        curr_token_id = BOS_ID
        dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
        dec_ids, attn_w = [], []
        curr_dec_idx = 0
        while (curr_token_id != EOS_ID and curr_dec_idx <= self.max_tgt_len):
            prev_y = self.embedding_mat(dec_input_var)
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            attn_w.append(decoder_attention.data)
            dis_o, dis_dec_h, dis_a = self.decoder(prev_y, dis_dec_h, dis_enc_o)
            #print(decoder_output.shape, dis_o.shape)
            s0_p = torch.cat( [decoder_output.unsqueeze(-1), dis_o.unsqueeze(-1)], dim=-1)
            gold_p = decoder_output + torch.log(cur_m[0][0])
            dis_p = dis_o + torch.log( cur_m[0][1] )
            s0_tmp = torch.cat( [gold_p.unsqueeze(-1), dis_p.unsqueeze(-1)], dim=-1)
            
            l0_p = F.log_softmax( s0_tmp, dim=-1 )
            l1_p = s0_p + l0_p
            s1_p = s0_p + alpha * l0_p
            l1_p = F.softmax(l1_p, dim=-1)
            s1_p = F.softmax(s1_p, dim=1)
            lp_ = torch.log(s1_p[:, :, 0] + 1e-8)
            _, p_id = lp_.data.topk(k=1, dim=-1)
            p_id, l1_p, cur_m = p_id.cuda().long(), l1_p.cuda(), cur_m.cuda()
            try:
                cur_m = cuda_if_gpu( torch.FloatTensor([l1_p[:, :, 0][0][w], l1_p[:, :, 1][0][w]]) )
                cur_m = cur_m.unsqueeze(0)
                cur_m /= torch.sum(cur_m, dim=-1).view(-1, 1)
            except:
                cur_m = cur_m
            #print(cur_m)
            curr_token_id = p_id[0][0]
            dec_ids.append(curr_token_id)
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
            curr_dec_idx += 1
        #exit()
        return dec_ids, attn_w

    def predict(self, input_var):

        # Embedding lookup
        encoder_input_embedded = self.embedding_lookup(input_var)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        # Decode
        dec_ids, attn_w = [], []
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
        dec_hidden = encoder_hidden[:1]  # 1 x B x enc_dim
        dec_probs = []
        while (curr_token_id != EOS_ID and curr_dec_idx <= self.max_tgt_len):
            prev_y = self.embedding_mat(dec_input_var)
            # prev_y = self.dropout(prev_y)

            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            attn_w.append(decoder_attention.data)

            topval, topidx = decoder_output.data.topk(1)
            curr_token_id = topidx[0][0]
            curr_token_prob = topval[0][0]
            dec_probs.append( curr_token_prob )
            dec_ids.append(curr_token_id)
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
            #print(topval)
            #exit()
            curr_dec_idx += 1

        return dec_ids, (attn_w, dec_probs)

    def predict_beam(self, input_var, beam_size=5, length_normalization_factor=0.0, length_normalization_const=5.):
        from components.model.Beam import Sequence, TopN
        partial_sequences = TopN(beam_size) 
        complete_sequences = TopN(beam_size)
        # Embedding lookup
        encoder_input_embedded = self.embedding_lookup(input_var)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        curr_token_id = BOS_ID
        dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
        dec_hidden = encoder_hidden[:1]  # 1 x B x enc_dim
        prev_y = self.embedding_mat(dec_input_var)
        decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs) # decode the logprobs, words

        logprobs, words = decoder_output.data.topk( beam_size )
        logprobs, words = logprobs[0], words[0]
        for k in range(beam_size):
                seq = Sequence(
                    output=[curr_token_id] + [words[k]],
                    state=dec_hidden,
                    logprob=logprobs[k],
                    score=logprobs[k],
                    attention=None)
                partial_sequences.push(seq)
        for i in range(self.max_tgt_len):
            partial_sequences_list = partial_sequences.extract()
            seqs = [x.output for x in partial_sequences_list]
            partial_sequences.reset()
            flattened_partial = [s for s in partial_sequences_list]
            input_feed = [c.output[-1] for c in flattened_partial]
            state_feed = [c.state for c in flattened_partial]
            # print(input_feed)
            if len(input_feed) == 0:
                break
            words_, logprobs_, states_ = [], [], []
            for input_feed_, state_feed_ in zip(input_feed, state_feed):
                dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([input_feed_])))
                prev_y = self.embedding_mat(dec_input_var)
                decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, state_feed_, encoder_outputs)
                logprobs, words = decoder_output.data.topk( beam_size + 1 )
                logprobs, words = logprobs[0], words[0]
                # print("dec_input_var, words", words )
                words_.append(words), logprobs_.append(logprobs), states_.append( dec_hidden )
            idx = 0
            # print(len(seqs), words_)
            for partial in partial_sequences_list:
                dec_hidden = states_[idx]
                attention = None
                k, num_hyp = 0, 0
                while num_hyp < beam_size:
                    w = words_[idx].data[k]
                    # print("w", w.data.item())
                    output = partial.output + [w]
                    # print(idx, "output", output, w.data.item() == EOS_ID)
                    logprob = partial.logprob + logprobs_[idx][k]
                    score = logprob
                    k += 1
                    num_hyp += 1
                    if w.data.item() == EOS_ID:
                            if length_normalization_factor > 0:
                                L = length_normalization_const
                                length_penalty = (L + len(output)) / (L + 1)
                                score /= length_penalty ** length_normalization_factor
                            beam = Sequence(output, dec_hidden,
                                            logprob, score, attention)
                            complete_sequences.push(beam)
                            num_hyp -= 1  # we can fit another hypotheses as this one is over
                    else:
                        beam = Sequence(output, dec_hidden, logprob, score, attention)
                        partial_sequences.push(beam)
                idx += 1
        seqs = complete_sequences.extract(sort=True)
        dec_ids, score = [], []
        for seq in seqs:
            dec_ids.append( seq.output[1:] )
            score.append( seq.score.data.item() )
            # print(seq.output, seq.score, seq.logprob)
        # print(dec_ids, score)
        # exit()
        return dec_ids, score
component = E2EMLPModel
