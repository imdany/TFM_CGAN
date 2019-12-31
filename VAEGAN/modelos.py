# Bloques convolucionales genericos
def conv_block(x, filters, leaky=True, transpose=False, name=''):
    conv = Conv2DTranspose if transpose else Conv2D
    activation = LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
    layers = [
        conv(filters, 3, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name=name + 'conv'),
        BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name=name + 'bn'),
        activation
    ]
    if x is None:
        return layers
    for layer in layers:
        x = layer(x)
    return x

def create_encoder():
    x = Input(shape=image_shape, name='enc_input')

    y = conv_block(x, 64,  name='enc_b_1_')
    y = conv_block(y, 128, name='enc_b_2_')
    y = conv_block(y, 256, name='enc_b_3_')
    y = Flatten()(y)
    y = Dense(n_encoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='enc_h_dense')(y)
    y = BatchNormalization(name='enc_h_bn')(y)
    y = LeakyReLU(leaky_relu_alpha)(y)

    z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='he_uniform')(y)
    z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='he_uniform')(y)

    return Model(x, [z_mean, z_log_var], name='encoder')


def create_decoder():
    x = Input(shape=(latent_dim,), name='enc_input')

    y = Dense(n_decoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dec_h_dense')(x)
    y = BatchNormalization(name='dec_h_bn')(y)
    y = LeakyReLU(leaky_relu_alpha)(y)
    y = Reshape(decode_from_shape)(y)
    y = *conv_block(y, 256, transpose=True, name='dec_b_1_')
    y = *conv_block(y, 128, transpose=True, name='dec_b_2_')
    y = *conv_block(y, 64,  transpose=True, name='dec_b_3_')
    y = Conv2D(n_channels, 5, activation='tanh', padding='same', kernel_regularizer=l2(wdecay),
               kernel_initializer='he_uniform', name='dec_output')

    return Model(x, y, name='decoder')


def create_discriminator():
    x = Input(shape=image_shape, name='dis_input')

    layers = [
        Conv2D(32, 5, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform',
                name='dis_blk_1_conv'),
        LeakyReLU(leaky_relu_alpha),
        *conv_block(None, 64,  leaky=True, name='dis_b_2_'),
        *conv_block(None, 128, leaky=True, name='dis_b_3_'),
        *conv_block(None, 256, leaky=True, name='dis_b_4_'),
        Flatten(),
        Dense(n_discriminator, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_dense'),
        BatchNormalization(name='dis_bn'),
        LeakyReLU(leaky_relu_alpha),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform',
                name='dis_output')
    ]

    y = x
    y_feat = None
    for i, layer in enumerate(layers, 1):
        y = layer(y)
        # Mapeo de las features al nivel correspondiente
        if i == recon_depth:
            y_feat = y

    return Model(x, [y, y_feat], name='discriminator')