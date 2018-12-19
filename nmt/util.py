import tensorflow as tf


def batching_scheme(batch_size, bucket_scale, max_length):
    bucket_edges = []
    b = 8
    while b < max_length:
        bucket_edges.append(b)
        b = max(b + 1, int(b * bucket_scale))

    def max_bucket_size(edge):
        return batch_size // edge

    bucket_sizes = [min(max_bucket_size(source_edge), max_bucket_size(target_edge))
                    for source_edge in bucket_edges for target_edge in bucket_edges] + [max_bucket_size(max_length)]
    bucket_edges = [(source_edge << 16) + target_edge
                    for source_edge in bucket_edges for target_edge in bucket_edges]
    print(bucket_edges)
    return {"boundaries": bucket_edges, "batch_sizes": bucket_sizes, "max_length": max_length}


def get_padded_shapes(dataset):
    """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.
    Args:
      dataset: The dataset that will be batched with padding.
    Returns:
      The same structure as ``dataset.output_shapes`` containing the padded
      shapes.
    """
    return tf.contrib.framework.nest.map_structure(lambda shape: shape.as_list(), dataset.output_shapes)


def batch_dataset(batch_size, padded_shapes=None, padding_values=None):
    """Transformation that batches a dataset.
    Args:
      batch_size: The batch size.
      padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
        are automatically inferred from the dataset output shapes.
    Returns:
      A ``tf.data.Dataset`` transformation.
    """
    return lambda dataset: dataset.padded_batch(batch_size,
                                                padded_shapes=padded_shapes or get_padded_shapes(dataset),
                                                padding_values=padding_values)


def batch_parallel_dataset(batch_size,
                           batch_type="examples",
                           batch_multiplier=1,
                           bucket_width=None,
                           padded_shapes=None,
                           padding_values=None,
                           features_length_fn=None,
                           labels_length_fn=None):
    """Transformation that batches a parallel dataset.
    This implements an example-based and a token-based batching strategy
    with optional bucketing of sequences.
    Bucketing makes the batches contain sequences of similar lengths to optimize
    the training efficiency. For example, if :obj:`bucket_width` is 5, sequences
    will be organized by lengths:
    1 - 5 | 6 - 10 | 11 - 15 | ...
    where the assigned length is the maximum of the source and target lengths.
    Then each batch will only consider sequences from the same bucket.
    Args:
      batch_size: The batch size.
      batch_type: The training batching strategy to use: can be "examples" or
        "tokens".
      batch_multiplier: The batch size multiplier to prepare splitting accross
        replicated graph parts.
      bucket_width: The sequence length bucket width.
      padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
        are automatically inferred from the dataset output shapes.
      features_length_fn: A callable mapping features to a sequence length.
      labels_length_fn: A callable mapping labels to a sequence length.
    Returns:
      A ``tf.data.Dataset`` transformation.
    Raises:
      ValueError: if :obj:`batch_type` is not one of "examples" or "tokens".
    """
    batch_size = batch_size * batch_multiplier

    def _key_func(features, labels):
        features_length = features_length_fn(features) if features_length_fn is not None else None
        labels_length = labels_length_fn(labels) if labels_length_fn is not None else None
        # For multi inputs, apply bucketing on the target side or none at all.
        if isinstance(features_length, list):
            features_length = None
        bucket_id = tf.constant(0, dtype=tf.int32)
        if features_length is not None:
            bucket_id = tf.maximum(bucket_id, features_length // bucket_width)
        if labels_length is not None:
            bucket_id = tf.maximum(bucket_id, labels_length // bucket_width)
        return tf.cast(bucket_id, tf.int64)

    def _reduce_func(unused_key, dataset):
        return dataset.apply(batch_dataset(batch_size, padded_shapes=padded_shapes, padding_values=padding_values))

    def _window_size_func(key):
        if bucket_width > 1:
            key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = batch_size // (key * bucket_width)
        if batch_multiplier > 1:
            # Make the window size a multiple of batch_multiplier.
            size = size + batch_multiplier - size % batch_multiplier
        return tf.cast(tf.maximum(size, batch_multiplier), tf.int64)

    if bucket_width is None:
        return batch_dataset(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

    if batch_type == "examples":
        return tf.data.experimental.group_by_window(
            _key_func, _reduce_func, window_size=batch_size)
    elif batch_type == "tokens":
        return tf.data.experimental.group_by_window(
            _key_func, _reduce_func, window_size_func=_window_size_func)
    else:
        raise ValueError(
            "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))
