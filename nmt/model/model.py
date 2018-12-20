import math

import tensorflow as tf
from . import layers


def build_embeddings(size, dim):
    word_embeddings = tf.get_variable("embeddings", [size, dim], tf.float32,
                                      initializer=tf.glorot_uniform_initializer())
    return word_embeddings


def transformer_encoder(inputs, params):
    inputs, input_lengths = inputs["indices"], inputs["length"]
    with tf.variable_scope("source_embeddings"):
        word_embeddings = build_embeddings(params.source_vocabulary_size, params.model_dim)

    inputs = tf.nn.embedding_lookup(word_embeddings, inputs) * math.sqrt(params.model_dim)

    positional_encodings = layers.positional_encoding(tf.shape(inputs)[1], params.model_dim)
    inputs += positional_encodings

    inputs = tf.nn.dropout(inputs, 1.0 - params.dropout)

    # (batch_size, 1, source_len)
    padding_mask = tf.expand_dims(tf.sequence_mask(input_lengths), 1, name="padding_mask")

    current = inputs
    for layer in range(params.num_encoder_blocks):
        with tf.variable_scope("encoder_layer_{}".format(layer)):
            current = layers.encoder_block(current, params.num_heads, params.model_dim,
                                           params.hidden_size, dropout=params.dropout,
                                           mask=padding_mask)

    outputs = layers.layer_norm(current, name="encoder_outputs")

    return outputs


def transformer_decoder(encoder_outputs, source_lengths, inputs, params):
    inputs, input_lengths = inputs["indices"], inputs["length"]
    with tf.variable_scope("target_embeddings"):
        word_embeddings = build_embeddings(params.target_vocabulary_size, params.model_dim)

    inputs = tf.nn.embedding_lookup(word_embeddings, inputs) * math.sqrt(params.model_dim)

    positional_encodings = layers.positional_encoding(tf.shape(inputs)[1], params.model_dim)
    inputs += positional_encodings

    inputs = tf.nn.dropout(inputs, 1.0 - params.dropout)

    # (input_len, input_len)
    future_mask = tf.sequence_mask(tf.range(1, tf.shape(inputs)[1] + 1))
    # (batch_size, input_len, input_len)
    future_mask = tf.logical_and(future_mask, tf.expand_dims(tf.sequence_mask(input_lengths), 1),
                                 name="self_attention_mask")
    # (batch_size, 1, source_len)
    encoder_mask = tf.expand_dims(tf.sequence_mask(source_lengths), 1, name="encoder_mask")

    current = inputs
    for layer in range(params.num_decoder_blocks):
        with tf.variable_scope("decoder_layer_{}".format(layer)):
            current = layers.decoder_block(encoder_outputs, current, params.num_heads, params.model_dim,
                                           params.hidden_size, dropout=params.dropout, self_attention_mask=future_mask,
                                           encoder_decoder_mask=encoder_mask)

    current = layers.layer_norm(current, name="decoder_outputs")

    batch_size = tf.shape(inputs)[0]
    logits = tf.matmul(tf.reshape(current, [-1, params.model_dim]), word_embeddings, transpose_b=True)
    logits = tf.reshape(logits, [batch_size, -1, params.target_vocabulary_size], name="logits")
    return logits


def transformer_decoder_for_inference(inputs, states, params):
    step = tf.shape(inputs)[1]
    encoder_outputs= states["encoder_output"]
    source_lengths = states["encoder_length"]

    with tf.variable_scope("target_embeddings"):
        word_embeddings = build_embeddings(params.target_vocabulary_size, params.model_dim)

    inputs = tf.nn.embedding_lookup(word_embeddings, inputs) * math.sqrt(params.model_dim)

    positional_encodings = layers.positional_encoding(1, params.model_dim, offset=step)
    inputs += tf.expand_dims(positional_encodings, 0)

    encoder_mask = tf.expand_dims(tf.sequence_mask(source_lengths), 1)  # (batch_size, 1, source_len)

    previous_outputs = inputs
    current_output = inputs[:, -1:]
    for layer in range(params.num_decoder_blocks):
        layer_name = "decoder_layer_{}".format(layer)
        with tf.variable_scope(layer_name):
            current_output = layers.decoder_block(encoder_outputs, current_output, params.num_heads, params.model_dim,
                                                  params.hidden_size, cached_outputs=previous_outputs,
                                                  encoder_decoder_mask=encoder_mask)
        previous_outputs = states[layer_name]
        previous_outputs = tf.concat([previous_outputs, current_output], 1)
        states[layer_name] = previous_outputs

    current_output = layers.layer_norm(current_output, name="decoder_outputs")

    current_output = tf.squeeze(current_output, 1)
    logits = tf.matmul(current_output, word_embeddings, transpose_b=True, name="logits")
    return logits
