import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers as tflayers
import math


def _positional_encoding(length, channels, min_timescale=1.0, max_timescale=1.e4, offset=0):
    """
    Produces a timing signal from sinusoids.
    :param length: Sequence length for the signal.
    :param channels: Number of channels. The number of different frequencies will be channels / 2
    :param min_timescale:
    :param max_timescale:
    :param offset: Position to start at
    :return: Tensor of shape (length, channels)
    """
    position = tf.cast(tf.range(offset, offset + length), tf.float32)
    n_frequencies = channels // 2
    # Calculating exp is more efficient than pow, so we calculate timescale increments for exp
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (n_frequencies - 1)
    # Multiplication is more efficient than division, so we calculate inverse timescales and multiply instead of
    # timescales and divide
    inv_frequencies = min_timescale * tf.exp(tf.cast(tf.range(n_frequencies), tf.float32) * -log_timescale_increment)
    arguments = tf.expand_dims(position, 1) * tf.expand_dims(inv_frequencies, 0)
    # Paper says to alternate sin and cos, but that doesn't matter
    signal = tf.concat([tf.sin(arguments), tf.cos(arguments)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.identity(signal, name="positional_encodings")
    return signal


def positional_encoding(length, channels, max_timescale=10000.0, offset=0):
    position = tf.cast(tf.range(offset, offset + length), tf.float32)
    div_term = tf.exp(tf.cast(tf.range(0, channels, 2), tf.float32) * -(math.log(max_timescale) / channels))
    arguments = tf.expand_dims(position, 1) * div_term
    signal = tf.stack([tf.sin(arguments), tf.cos(arguments)], axis=2)
    signal = tf.reshape(signal, [length, channels], name="positional_encodings")
    return signal


def layer_norm(x, epsilon=1e-6, name=None, reuse=None):
    """
    Nobody knows what tensorflow's layer_norm does, so we use our own.
    :param x:
    :param epsilon:
    :param name:
    :param reuse:
    :return:
    """
    n_filters = shape_list(x)[-1]
    with tf.variable_scope(
            name, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable("layer_norm_scale", [n_filters], initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [n_filters], initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias


def norm_residual_sublayer(inputs, outputs, dropout=0.0):
    # Despite what the paper says, they always seem to normalize, apply, dropout, residual in that order
    outputs = tf.nn.dropout(outputs, 1.0 - dropout)
    return layer_norm(inputs + outputs)


def mask_logits(logits, mask, mask_value=-1e9):
    """
    Mask out some entries of logits, replacing them with (effectively) negative infinity.
    :param logits:
    :param mask: Zero entries will be masked out, ones will be left in. Supports broadcasting
    :param mask_value:
    :return:
    """
    # shapes = logits.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return logits + mask_value * (1 - mask)


def scaled_dot_product_attention(queries, keys, values, dropout=0.0, mask=None):
    """
    Calculate scaled dot product attention
    """
    scale_fac = tf.rsqrt(tf.cast(keys.shape.as_list()[-1], tf.float32))
    # (batch_size, <n_heads>, length_q, length_kv
    logits = tf.matmul(queries, keys, transpose_b=True) * scale_fac
    if mask is not None:
        if mask.shape.ndims == 3:  # mask is shape (batch_size, length_q, length_kv)
            mask = tf.expand_dims(mask, 1)  # allow broadcasting to heads

        logits = mask_logits(logits, mask)
    weights = tf.nn.softmax(logits, name="attention_weights")
    if dropout > 0.0:
        weights = tf.nn.dropout(weights, 1.0 - dropout)
    return tf.matmul(weights, values)


def multi_head_attention(queries, keys, values, num_heads, units, dropout=0.0, mask=None):
    assert units % num_heads == 0

    def split_heads(tensor):
        with tf.name_scope("split_heads"):
            old_shape = shape_list(tensor)
            last = old_shape[-1]
            tensor = tf.reshape(tensor, old_shape[:-1] + [num_heads, last // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])

    def join_heads(tensor):
        with tf.name_scope("join_heads"):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])
            old_shape = shape_list(tensor)
            a, b = old_shape[-2:]
            tensor = tf.reshape(tensor, old_shape[:-2] + [a * b])
            return tensor

    with arg_scope([tflayers.fully_connected], num_outputs=units, activation_fn=None,
                   weights_initializer=tf.glorot_uniform_initializer()):
        projected_queries = split_heads(tflayers.fully_connected(queries, scope="query_projection"))
        projected_keys = split_heads(tflayers.fully_connected(keys, scope="key_projection"))
        projected_values = split_heads(tflayers.fully_connected(values, scope="value_projection"))

    attention_output = scaled_dot_product_attention(projected_queries, projected_keys, projected_values, dropout, mask)
    attention_output = join_heads(attention_output)
    attention_output = tflayers.fully_connected(attention_output, units, None,
                                                weights_initializer=tf.glorot_uniform_initializer(),
                                                scope="output_projection")
    return attention_output


def feed_forward_net(inputs, hidden_size, dropout=0.0):
    output_size = inputs.shape.as_list()[-1]
    hidden = tflayers.fully_connected(inputs, hidden_size, activation_fn=tf.nn.relu,
                                      weights_initializer=tf.glorot_uniform_initializer(),
                                      scope="hidden_layer")
    hidden = tf.nn.dropout(hidden, 1.0 - dropout)
    output = tflayers.fully_connected(hidden, output_size, activation_fn=None,
                                      weights_initializer=tf.glorot_uniform_initializer(),
                                      scope="output_layer")
    return output


def encoder_block(inputs, num_heads, num_units, hidden_size, dropout=0.0, mask=None):
    with tf.variable_scope("self_attention_layer"):
        inputs = layer_norm(inputs)
        attention_output = multi_head_attention(inputs, inputs, inputs,
                                                num_heads, num_units, dropout, mask)
        if dropout > 0:
            attention_output = tf.nn.dropout(attention_output, 1.0 - dropout)
        attention_output += inputs

    with tf.variable_scope("feed_forward_layer"):
        attention_output = layer_norm(attention_output)
        output = feed_forward_net(attention_output, hidden_size, dropout)
        if dropout > 0:
            output = tf.nn.dropout(output, 1.0 - dropout)
        output += attention_output

    output = layer_norm(output)

    return output


def decoder_block(encoder_outputs, inputs, num_heads, num_units, hidden_size, dropout=0.0, self_attention_mask=None,
                  encoder_decoder_mask=None, cached_outputs=None):
    if cached_outputs is None:
        cached_outputs = inputs

    with tf.variable_scope("self_attention_layer"):
        inputs = layer_norm(inputs)
        attention_output = multi_head_attention(inputs, cached_outputs, cached_outputs,
                                                num_heads, num_units, dropout, self_attention_mask)
        if dropout > 0:
            attention_output = tf.nn.dropout(attention_output, 1.0 - dropout)
        attention_output += inputs

    with tf.variable_scope("encoder_decoder_attention_layer"):
        x = layer_norm(attention_output)
        attention_output = multi_head_attention(x, encoder_outputs, encoder_outputs,
                                                num_heads, num_units, dropout, encoder_decoder_mask)
        if dropout > 0:
            attention_output = tf.nn.dropout(attention_output, 1.0 - dropout)
        attention_output += x

    with tf.variable_scope("feed_forward_layer"):
        attention_output = layer_norm(attention_output)
        output = feed_forward_net(attention_output, hidden_size, dropout)

        if dropout > 0:
            output = tf.nn.dropout(output, 1.0 - dropout)
        output += attention_output

    output = layer_norm(output)

    return output


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
