import logging

#import keras.backend as K
#from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Activation, Embedding, Input
from tensorflow.keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import tensorflow as tf



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, maxlen, vocab):

    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / tf.cast(tf.keras.backend.epsilon() + tf.sqrt(tf.reduce_sum(tf.square(weight_matrix), axis=-1, keepdims=True)), tf.keras.backend.floatx())
        reg = tf.reduce_sum(tf.square(tf.matmul(w_n, tf.transpose(w_n)) - tf.eye(w_n.shape.as_list()[0])))
        return args.ortho_reg*reg

    vocab_size = len(vocab)

    ##### Inputs #####
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size, maxlen), dtype='int32', name='neg_input')

    ##### Construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    ##### Compute sentence representation #####
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    att_weights = Attention(name='att_weights')([e_w, y_s])
    z_s = WeightedSum()([e_w, att_weights])

    ##### Compute representations of negative instances #####
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)

    ##### Reconstruction #####
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
            W_regularizer=ortho_reg)(p_t)

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(inputs=[sentence_input, neg_input], outputs=loss)

    ### Word embedding and aspect embedding initialization ######
    if args.emb_path:
        from w2vEmbReader import W2VEmbReader as EmbReader
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        logger.info('Initializing word embedding matrix')
        #model.get_layer('word_emb').W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').W.get_value()))
        #아래 코드로 수정
        word_emb_layer = model.get_layer('word_emb')#수정한 코드
        weights = word_emb_layer.get_weights()[0]#수정한 코드
        new_weights = emb_reader.get_emb_matrix_given_vocab(vocab, weights)#수정한 코드
        word_emb_layer.set_weights([new_weights])#수정한 코드

        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        model.get_layer('aspect_emb').W.assign(emb_reader.get_aspect_matrix(args.aspect_size))#set_value를 assign으로 고침

    return model



    





