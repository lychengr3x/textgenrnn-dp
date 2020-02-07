from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import config as config
from tensorflow.compat.v1 import losses
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamOptimizer
from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianAverageQuery
from .AttentionWeightedAverage import AttentionWeightedAverage


def textgenrnn_model(num_classes, cfg, context_size=None,
                     weights_path=None,
                     dropout=0.0,
                     optimizer=Adam(lr=4e-3)):
    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''
    
    input = Input(shape=(cfg['max_length'],), name='input')
    embedded = Embedding(num_classes, cfg['dim_embeddings'],
                         input_length=cfg['max_length'],
                         name='embedding')(input)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer = embedded if i == 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn(cfg, i+1)(prev_layer))

    seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
    attention = AttentionWeightedAverage(name='attention')(seq_concat)
    output = Dense(num_classes, name='output', activation='softmax')(attention)

    if context_size is None:
        model = Model(inputs=[input], outputs=[output])
        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)

        # differential privacy 
        if "dp" in cfg:
            print("Implement Differential Privacy")
            # Compute vector of per-example loss rather than its mean over a minibatch.
            loss = CategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.NONE)
            lr = 4e-3 if "lr" not in cfg else cfg["lr"]
            l2_norm_clip = 1.0 if "l2_norm_clip" not in cfg else cfg["l2_norm_clip"]
            noise_multiplier = 1.1 if "noise_multiplier" not in cfg else cfg["noise_multiplier"]
            num_microbatches = 32 if "num_microbatches" not in cfg else cfg["num_microbatches"]
            dp_average_query = GaussianAverageQuery(l2_norm_clip, 
                                                    l2_norm_clip * noise_multiplier, 
                                                    num_microbatches)
            optimizer = DPAdamOptimizer(dp_sum_query=dp_average_query, 
                                        num_microbatches=num_microbatches, 
                                        learning_rate=lr)
        else:
            loss='categorical_crossentropy'

        model.compile(loss=loss, optimizer=optimizer)

    else:
        context_input = Input(
            shape=(context_size,), name='context_input')
        context_reshape = Reshape((context_size,),
                                  name='context_reshape')(context_input)
        merged = concatenate([attention, context_reshape], name='concat')
        main_output = Dense(num_classes, name='context_output',
                            activation='softmax')(merged)

        model = Model(inputs=[input, context_input],
                      outputs=[main_output, output])
        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      loss_weights=[0.8, 0.2])

    return model


'''
Create a new LSTM layer per parameters. Unfortunately,
each combination of parameters must be hardcoded.

The normal LSTMs use sigmoid recurrent activations
for parity with CuDNNLSTM:
https://github.com/keras-team/keras/issues/8860
'''

'''
FIXME
From TensorFlow 2 you do not need to specify CuDNNLSTM.
You can just use LSTM with no activation function and it will
automatically use the CuDNN version.
This part can probably be cleaned up.
'''

def new_rnn(cfg, layer_num):
    use_cudnnlstm = K.backend() == 'tensorflow' and len(config.get_visible_devices('GPU')) > 0
    if use_cudnnlstm:
        if cfg['rnn_bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                           return_sequences=True),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(cfg['rnn_size'],
                         return_sequences=True,
                         name='rnn_{}'.format(layer_num))
    else:
        if cfg['rnn_bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                      return_sequences=True,
                                      recurrent_activation='sigmoid'),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(cfg['rnn_size'],
                    return_sequences=True,
                    recurrent_activation='sigmoid',
                    name='rnn_{}'.format(layer_num))
