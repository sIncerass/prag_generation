from __future__ import absolute_import, division
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import keras
from keras.optimizers import *
from keras.models import Model, Sequential
from keras.regularizers import L1L2
from keras.layers import *
import pickle as pk
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """
    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        #self.init = initializations.get('uniform') 
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

def finetuning_callbacks(checkpoint_path, patience, verbose):
    """ Callbacks for model training.
    # Arguments:
        checkpoint_path: Where weight checkpoints should be saved.
        patience: Number of epochs with no improvement after which
            training will be stopped.
    # Returns:
        Array with training callbacks that can be passed straight into
        model.fit() or similar.
    """
    cb_verbose = (verbose >= 2)
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path,
                                   save_best_only=True, verbose=cb_verbose)
    earlystop = EarlyStopping(monitor='val_loss', patience=patience,
                              verbose=cb_verbose)
    return [checkpointer, earlystop]

# 512 dim gru
def reconstructor(w2v_dim, pretrain_w2v, nb_tokens, max_src_len, mr_value_num, MR_FIELDS, ra=False, embed_l2=1E-6):
    model_input = Input(shape=(max_src_len,), dtype='int32')
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    if pretrain_w2v is None:
        embed = Embedding(input_dim=nb_tokens, output_dim=w2v_dim, mask_zero=True, input_length=max_src_len, embeddings_regularizer=embed_reg)
    else:
        embed = Embedding(input_dim=nb_tokens, output_dim=w2v_dim, mask_zero=True, input_length=max_src_len, weights=[pretrain_w2v], embeddings_regularizer=embed_reg, trainable=True)
    embed_x = embed(model_input)
    #embed = Activation('tanh')(embed)
    context_emb = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(embed_x)
    outputs = []
    for mr_model in MR_FIELDS:
        x = AttentionWeightedAverage(name='attlayer_%s' % mr_model.split()[0], return_attention=ra)(context_emb)
        outputs.append( Dense(mr_value_num[mr_model], activation='softmax', name='softmax_%s' % mr_model.split()[0])(x) )
    return Model(inputs=[model_input], outputs=outputs, name="reconstruct")

MR_FIELDS = ["name", "familyFriendly", "eatType", "food", "priceRange", "near", "area", "customer rating"]
w2v_dim = 128
patience = 5
opt_para = 'sgd'
max_src_len, max_tgt_len = 50, 50
mr_value_id2vocab, mr_value_vocab2id, mr_value_num = pk.load(open("mr_value.pkl", "rb"))
(recs, recs_lexical) = pk.load(open("dev_recs.pkl", "rb"))
(train, train_lexical), (val, val_lexical) = pk.load(open("train.pkl", "rb")), pk.load(open("dev.pkl", "rb"))
embed_pretain = np.array(pk.load(open("w2v.%dd.pkl" % w2v_dim, "rb")))
nb_tokens = embed_pretain.shape[0]
checkpoint_weight_path = "ckpt/reconstruct_%d.h5" % w2v_dim
mr_value_num['name'] = 2
model = reconstructor(w2v_dim, np.array(embed_pretain), nb_tokens, max_src_len, mr_value_num, MR_FIELDS)
model.load_weights(checkpoint_weight_path)
#model.summary()
epochs = 50
lossFc = 'sparse_categorical_crossentropy'
if opt_para == 'adam':
    opt = Adam(clipnorm=5, lr=1e-3)
else:
    opt = SGD(lr=0.1)
model.compile(loss=lossFc, optimizer=opt, metrics=['accuracy'])
xx = [np.array(x) for x in recs[1]]
print(recs[1], len(recs[0]))
print(xx[0].shape, xx[1].shape, xx[2].shape, xx[3].shape, xx[4].shape, xx[5].shape, xx[6].shape, xx[7].shape)
#score = model.evaluate(np.array(val[0]), val[1], verbose=True)
losses = []
import time
start = time.time()
for recs_idx, recs_ref in enumerate( recs[0] ):
    recs_x = np.array([recs_ref])
    recs_y = [ recs[1][0][recs_idx], recs[1][1][recs_idx], recs[1][2][recs_idx], recs[1][3][recs_idx], recs[1][4][recs_idx], recs[1][5][recs_idx], recs[1][6][recs_idx], recs[1][7][recs_idx]  ]
    recs_y = [np.array([x]) for x in recs_y]
    score = model.evaluate(recs_x, recs_y, verbose=False)
    losses.append(score[0])
    if recs_idx % 100 == 0:
        print(recs_idx, time.time() - start)
    #print(model.metrics_names, score)
    #break
pk.dump(losses, open("dev_recs_loss.pkl", "wb"))
exit()
#score = model.evaluate(np.array(recs[0]), recs[1], verbose=True)
print(model.metrics_names, score)
print(score.history)
print('Test score: ', score[0])    #Loss on test
print('Test accuracy: ', score[1])
