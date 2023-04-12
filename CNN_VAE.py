import tensorflow as tf

import tensorflow_probability as tfp


class cnn_vae:
        
    encoded_dim = 32                   # 潜在空間の次元数
    seq_length = None        # 時間軸上の要素数
    input_dim = None         # 入力データにおける各時刻のベクトルの次元数
    output_dim = None        # 出力データにおける各時刻のベクトルの次元数
    lstm_dim = 1024                    # LSTM層のノード数

    def __init__(self, x_all, y_all):
        self.seq_length = x_all.shape[1]
        self.input_dim = x_all.shape[2]
        self.output_dim = y_all.shape[2]
        
        
    # VAEに用いる事前分布を定義
    def make_prior(self):
        tfd = tfp.distributions
        prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(self.encoded_dim), scale=1), 
            reinterpreted_batch_ndims=1)
        return prior


    # エンコーダを構築
    def make_encoder(self, prior):
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.LSTM(self.lstm_dim, 
                                        input_shape=(self.seq_length, self.input_dim),
                                        use_bias=True, activation="tanh", 
                                        return_sequences=False))

        encoder.add(tf.keras.layers.Dense(
            tfp.layers.MultivariateNormalTriL.params_size(self.encoded_dim), 
            activation=None))

        encoder.add(tfp.layers.MultivariateNormalTriL(
            self.encoded_dim, 
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                prior, weight=0.001)))

        return encoder


    # デコーダを構築
    def make_decoder(self):
        decoder = tf.keras.Sequential()
        decoder.add(tf.keras.layers.RepeatVector(self.seq_length, 
                                                input_dim=self.encoded_dim))
        decoder.add(tf.keras.layers.LSTM(self.lstm_dim, use_bias=True, 
                                        activation="tanh", 
                                        return_sequences=True))
        decoder.add(tf.keras.layers.Dense(self.output_dim, use_bias=True, 
                                            activation="softmax"))
        return decoder


    # エンコーダとデコーダを構築し、それらを結合したモデルを構築する
    # （入力：エンコーダの入力、
    # 　出力：エンコーダの出力をデコーダに入力して得られる出力）
    def make_model(self):

        encoder = self.make_encoder(self.make_prior())
        decoder = self.make_decoder()
        vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                    loss="categorical_crossentropy", metrics="categorical_accuracy")
        return vae