import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib import distribute
from tqdm import tqdm

from nmt import model
from nmt.beam_search import beam_search
from nmt.bleu import compute_bleu, bleu_tokenize
from nmt.util import batch_parallel_dataset

import os
import re

bpe_re = re.compile(r"(@@ )|(@@ ?$)")


def get_distribution_strategy(params):
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(gpus) > 1 and params.max_gpus > 1:
        actual_devices = min(len(gpus), params.max_gpus)
        params.using_gpu = True
        strategy = distribute.MirroredStrategy(num_gpus=actual_devices)
    elif len(gpus) > 0 and params.max_gpus > 0:
        actual_devices = 1
        params.using_gpu = True
        strategy = distribute.OneDeviceStrategy("/GPU:0")
    else:
        actual_devices = 1
        params.using_gpu = False
        strategy = distribute.OneDeviceStrategy("/CPU:0")
    params.actual_devices = actual_devices
    return strategy


def build_vocabulary(params):
    source_vocabulary = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=params.source_vocabulary_file,
        num_oov_buckets=1,
        key_column_index=0,
        delimiter=" "
    )
    source_vocabulary.init.run()

    vocabulary_size = source_vocabulary.size().eval()  # includes oov
    params.source_end_id = vocabulary_size
    params.source_pad_id = vocabulary_size + 1
    params.source_vocabulary_size = vocabulary_size + 2

    target_vocabulary = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=params.target_vocabulary_file,
        num_oov_buckets=1,
        key_column_index=0,
        delimiter=" "
    )
    target_vocabulary.init.run()

    vocabulary_size = target_vocabulary.size().eval()  # includes oov
    params.target_end_id = vocabulary_size
    params.target_pad_id = vocabulary_size + 1
    params.target_vocabulary_size = vocabulary_size + 2
    
    return source_vocabulary, target_vocabulary


def build_input_pipeline(filename, vocabulary, params, source=False):
    dataset = tf.data.TextLineDataset(filename)

    def process(sample):
        sample = tf.string_split([sample], " ")
        word_indices = vocabulary.lookup(sample.values)
        end_id = params.source_end_id if source else params.target_end_id
        word_indices = tf.pad(word_indices, [[0, 1]] if source else [[1, 1]], constant_values=end_id)
        sample_length = tf.size(word_indices) if source else tf.size(word_indices) - 1
        return {"indices": word_indices, "length": sample_length}

    dataset = dataset.map(process, params.num_threads)
    return dataset


def build_dataset(source, target, vocabularies, params, shuffle_repeat=True):
    if target is None:
        target = source

    dataset = tf.data.Dataset.zip((build_input_pipeline(source, vocabularies[0], params, True),
                                   build_input_pipeline(target, vocabularies[1], params)))

    dataset = dataset.filter(lambda x, y: (x["length"] <= params.max_example_length) &
                                          (y["length"] <= params.max_example_length))

    if shuffle_repeat:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(params.shuffle_buffer_size))

    dataset = dataset.apply(batch_parallel_dataset(
        params.batch_size, "tokens", 1, 2,
        ({"indices": [None], "length": []},) * 2,
        ({"indices": params.source_pad_id, "length": 0}, {"indices": params.target_pad_id, "length": 0}),
        lambda x: x["length"],
        lambda x: x["length"]
    ))

    if shuffle_repeat:
        dataset = dataset.shuffle(50)

    dataset = dataset.prefetch(1)

    return dataset


def build_model(source, params, target=None):
    encoder_outputs = model.transformer_encoder(source, params)
    if target is None:  # if no labels are given, we must be in inference
        input_indices = source["indices"]

        def symbols_to_logits_fn(decoder_inputs, i, states):
            return model.transformer_decoder_for_inference(decoder_inputs, states, params), states

        # TODO: should be agnostic to decoder architecture
        states = {"decoder_layer_{}".format(i): tf.zeros([tf.shape(encoder_outputs)[0], 0, params.model_dim])
                  for i in range(params.num_decoder_blocks)}
        states["encoder_output"] = encoder_outputs
        states["encoder_length"] = source["length"]

        batch_size = tf.shape(input_indices)[0]

        beams, probs = beam_search(symbols_to_logits_fn,
                                   params.target_end_id * tf.ones([batch_size], dtype=tf.int32),
                                   params.beam_size,
                                   tf.shape(input_indices)[1] + 50,
                                   params.target_vocabulary_size,
                                   params.length_penalty,
                                   states=states,
                                   eos_id=params.target_end_id,
                                   stop_early=True)
        batch_size = tf.shape(beams)[0]
        return tf.gather_nd(beams, tf.stack([tf.range(batch_size), tf.argmax(probs, 1, output_type=tf.int32)], 1))
    else:
        decoder_inputs = target["indices"][:, :-1]
        decoder_inputs = tf.identity(decoder_inputs, name="decoder_inputs")
        decoder_inputs = {"indices": decoder_inputs, "length": target["length"]}
        return model.transformer_decoder(encoder_outputs, source["length"], decoder_inputs, params)


def build_loss(logits, target, params):
    target, label_lengths = target["indices"], target["length"]
    target = target[:, 1:]

    one_hot_labels = tf.one_hot(target, params.target_vocabulary_size, dtype=tf.float32, name="one_hot_labels")
    smoothed_labels = tf.add(one_hot_labels * (1.0 - params.label_smoothing),
                             params.label_smoothing / params.target_vocabulary_size,
                             name="smoothed_labels")
    mask = tf.sequence_mask(label_lengths, dtype=tf.float32, name="sequence_weights")
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_labels, logits=logits, name="raw_loss")
    loss = tf.multiply(loss, mask, name="masked_loss")
    return loss


def build_optimizer(params):
    step = tf.cast(tf.train.get_global_step(), tf.float32)
    learning_rate = (params.model_dim ** -0.5) * tf.minimum(tf.rsqrt(step),
                                                            step * (params.warmup_steps ** -1.5))

    # learning_rate *= 2.5
    # learning_rate = 1.e-3

    tf.summary.scalar("learning rate", learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate, params.beta1, params.beta2, params.adam_epsilon)
    return opt


def train(params):
    train_dir = os.path.join(params.train_dir, params.model_name)
    event_dir = os.path.join(params.event_dir, params.model_name)

    for folder in [train_dir, event_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    with open(params.source_vocabulary_file) as fh:
        source_lookup = [line.split(" ")[0] for line in fh]

    source_lookup.extend(["<oov>", "<end>", "<pad>"])

    with open(params.target_vocabulary_file) as fh:
        target_lookup = [line.split(" ")[0] for line in fh]

    target_lookup.extend(["<oov>", "<end>", "<pad>"])

    lookup = (source_lookup, target_lookup)

    with tf.Graph().as_default():
        # Profiling
        # builder = tf.profiler.ProfileOptionBuilder
        # opts = builder(builder.time_and_memory()).order_by('micros').build()
        # opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()

        global_step = tf.train.create_global_step()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        # sess_config.gpu_options.allow_growth = True
        # with tf.contrib.tfprof.ProfileContext("../tmp/trace", trace_steps=[55, 65, 75], dump_steps=[75]) as pctx,\
        #         tf.Session(config=sess_config).as_default() as session:

        # Debugger
        # session = tf.Session(config=sess_config)
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        # with session:

        with tf.Session(config=sess_config).as_default() as session:
            # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.
            # pctx.add_auto_profiling('op', opts, [55, 65, 75])
            # Run online profiling with 'scope' view and 'opts2' options at step 20.
            # pctx.add_auto_profiling('scope', opts2, [75])

            vocabularies = build_vocabulary(params)

            with tf.name_scope("train"):
                optimizer = build_optimizer(params)

                with tf.device("/CPU:0"):
                    dataset = build_dataset(params.training_source, params.training_target, vocabularies, params)
                train_iterator = dataset.make_initializable_iterator()
                session.run(train_iterator.initializer)
                source, target = train_iterator.get_next()

                with tf.variable_scope("model"):
                    outputs = build_model(source, params, target)
                train_loss = build_loss(outputs, target, params)  # (?, ?)

                if params.accumulate_gradients == 1:
                    mean_loss = tf.reduce_sum(train_loss) / tf.cast(tf.reduce_sum(target["length"]), tf.float32)
                    train_op = optimizer.minimize(mean_loss, global_step, tf.trainable_variables("model"))
                    accumulate_op = zero_op = None

                else:
                    # Accumulate gradients to simulate multi-GPU training
                    trainable_variables = tf.trainable_variables("model")
                    gradients = optimizer.compute_gradients(train_loss)

                    gradient_accumulators = {var.name: tf.get_variable(var.name.split(":")[0] + "/grad_acc", var.shape,
                                                                       initializer=tf.zeros_initializer(), trainable=False)
                                             for var in trainable_variables}
                    loss_accumulator = tf.get_variable("loss_acc", [], tf.float32, tf.zeros_initializer(),
                                                       trainable=False)
                    word_accumulator = tf.get_variable("word_acc", [], tf.float32, tf.zeros_initializer(),
                                                       trainable=False)
                    loss_acc_op = tf.assign_add(loss_accumulator, tf.reduce_sum(train_loss))
                    word_acc_op = tf.assign_add(word_accumulator, tf.cast(tf.reduce_sum(target["length"]), tf.float32))
                    accumulate_op = tf.group([tf.assign_add(gradient_accumulators[var.name],
                                                            tf.reduce_sum(tf.reshape(grad, [-1] + var.shape.as_list()), 0))
                                              for grad, var in gradients] + [loss_acc_op, word_acc_op])
                    train_op = optimizer.apply_gradients([(gradient_accumulators[var.name] / word_accumulator, var)
                                                          for _, var in gradients], global_step=global_step)
                    with tf.control_dependencies([train_op]):
                        zero_op = tf.group([tf.assign(var, tf.zeros_like(var)) for var in gradient_accumulators.values()] +
                                           [tf.assign(loss_accumulator, 0.0), tf.assign(word_accumulator, 0)])
                    mean_loss = loss_accumulator / word_accumulator

                perplexity = tf.exp(mean_loss)
                tf.summary.scalar("loss", mean_loss)
                tf.summary.scalar("perplexity", tf.exp(mean_loss))

            # with tf.name_scope("dev"), tf.device("/GPU:0" if params.using_gpu else "/CPU:0"):
            with tf.name_scope("dev"):
                with tf.device("/CPU:0"):
                    dataset = build_dataset(params.development_source, params.development_target, vocabularies, params,
                                            shuffle_repeat=False)
                dev_iterator = dataset.make_initializable_iterator()
                session.run(dev_iterator.initializer)
                source, target = dev_iterator.get_next()
                with tf.name_scope("forced"), tf.variable_scope("model", reuse=True):
                    dev_outputs_forced = build_model(source, params, target)

                with tf.name_scope("search"), tf.variable_scope("model", reuse=True):
                    dev_outputs_search = build_model(source, params)

                dev_loss = build_loss(dev_outputs_forced, target, params)

            summary_op = tf.summary.merge_all(scope="train")
            summary_writer = tf.summary.FileWriter(event_dir, session.graph)

            session.run(tf.global_variables_initializer())
            summary_writer.flush()

            checkpoint_writer = tf.train.Saver(max_to_keep=5)

            if os.path.exists(os.path.join(train_dir, "checkpoint")):
                checkpoint_writer.restore(session, tf.train.latest_checkpoint(train_dir))
            step = global_step.eval() * params.accumulate_gradients

            def checkpoint(step):
                summaries = evaluate_dev_set(session, dev_outputs_search, dev_outputs_forced, dev_loss, dev_iterator,
                                             target, lookup, params)
                for k, v in summaries.items():
                    summary_writer.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="dev/" + k, simple_value=v)]), step)
                summary_writer.flush()
                checkpoint_filename = os.path.join(train_dir, "{}_{}.ckpt".format(params.model_name, step))
                checkpoint_writer.save(session, checkpoint_filename, write_meta_graph=False)

            # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

            total_training_steps = params.training_steps * params.accumulate_gradients

            with tqdm(total=total_training_steps, initial=step, unit="step", postfix="??") as pbar:
                if params.accumulate_gradients == 1:
                    ema_perplexity = None
                    for step in range(step // params.accumulate_gradients + 1, params.training_steps + 1):
                        _, summary_string, ppl = session.run([train_op, summary_op, perplexity])

                        if np.isnan(ppl):
                            raise ValueError("NaN loss encountered in step {}".format(step))

                        if ema_perplexity is None:
                            ema_perplexity = ppl
                        else:
                            ema_perplexity = 0.6 * ema_perplexity + 0.4 * ppl

                        pbar.set_postfix_str("PPL: {:.1f}".format(ema_perplexity))
                        pbar.update()

                        if step % params.log_interval == 0 or step < params.log_interval:
                            summary_writer.add_summary(summary_string, step)
                            summary_writer.flush()

                        if step % params.checkpoint_interval == 0:
                            checkpoint(step)

                else:
                    for step in range(step // params.accumulate_gradients + 1, params.training_steps + 1):
                        for _ in range(params.accumulate_gradients):
                            session.run([accumulate_op])
                            pbar.update()

                        _, summary_string, ppl, _ = session.run([train_op, summary_op, perplexity, zero_op])

                        if np.isnan(ppl):
                            raise ValueError("NaN loss encountered in step {}".format(step))

                        pbar.set_postfix_str("PPL: {:.1f}".format(ppl))

                        if step % params.log_interval == 0 or step < params.log_interval:
                            summary_writer.add_summary(summary_string, step)
                            summary_writer.flush()

                        if step % params.checkpoint_interval == 0:
                            checkpoint(step)

            checkpoint(step)


def evaluate_dev_set(session, outputs_search, outputs_forced, loss, iterator, labels, lookup, params):
    label_indices = labels["indices"]
    label_lengths = labels["length"]

    source_lookup, target_lookup = lookup

    label_strings = []
    forced_predictions = []
    predictions = []
    total_loss = 0.0
    total_words = 0

    def assemble_string(lookup, indices):
        words = []
        for index in indices:
            if index >= len(lookup):
                continue
            elif lookup[index] == "<end>":
                break
            else:
                words.append(lookup[index])
        return bpe_re.sub("", " ".join(words))

    session.run(iterator.initializer)
    with tqdm(total=params.development_samples, postfix="evaluation", unit="word") as pbar:
        while True:
            try:
                pred, pred2, lab, labl, lo = session.run([outputs_search, outputs_forced, label_indices, label_lengths, loss])
                predictions.extend(assemble_string(target_lookup, p[1:]) for p in pred)
                forced_predictions.extend(assemble_string(target_lookup, p) for p in np.argmax(pred2, -1))
                label_strings.extend(assemble_string(target_lookup, l[1:]) for l in lab)
                total_loss += np.sum(lo)
                n_words = np.sum(labl)
                total_words += n_words
                pbar.update(n_words)
            except tf.errors.OutOfRangeError:
                break

    mean_loss = total_loss / total_words
    mean_perplexity = np.exp(mean_loss)
    bleu = compute_bleu([bleu_tokenize(ref) for ref in label_strings],
                        [bleu_tokenize(pred) for pred in predictions])

    output_dir = os.path.join(params.output_dir, params.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "beam_output"), "w") as fh:
        fh.writelines(p + "\n" for p in predictions)

    with open(os.path.join(output_dir, "eval_output"), "w") as fh:
        fh.writelines(p + "\n" for p in forced_predictions)

    with open(os.path.join(output_dir, "verification_labels"), "w") as fh:
        fh.writelines(p + "\n" for p in label_strings)

    summaries = {
        "loss": mean_loss,
        "perplexity": mean_perplexity,
        "BLEU": bleu
    }
    return summaries


def test(params):
    train_dir = os.path.join(params.train_dir, params.model_name)

    with open(params.vocabulary_file) as fh:
        lookup = [line.split(" ")[0] for line in fh]

    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config).as_default() as session:
            vocabulary = build_vocabulary(params)

            with tf.name_scope("test"):
                with tf.device("/CPU:0"):
                    dataset = build_dataset(params.test_source, params.test_target, vocabulary, params,
                                            shuffle_repeat=False)
                test_iterator = dataset.make_initializable_iterator()
                session.run(test_iterator.initializer)
                inputs, labels = test_iterator.get_next()
                with tf.name_scope("search"), tf.variable_scope("model"):
                    test_outputs = build_model(inputs, params)

                with tf.name_scope("forced"), tf.variable_scope("model", reuse=True):
                    test_forced = build_model(inputs, params, labels)

                test_loss = build_loss(test_forced, labels, params)

            session.run(tf.global_variables_initializer())
            checkpoint_writer = tf.train.Saver()
            checkpoint_writer.restore(session, tf.train.latest_checkpoint(train_dir))

            label_indices = labels["indices"]
            label_lengths = labels["length"]

            predictions = []
            forced_predictions = []
            label_strings = []
            total_loss = 0.0
            total_words = 0

            def assemble_string(indices):
                words = []
                for index in indices:
                    if index >= len(lookup):
                        continue
                    else:
                        words.append(lookup[index])
                return bpe_re.sub("", " ".join(words))

            with tqdm(total=params.test_samples, unit="words") as pbar:
                while True:
                    try:
                        pred, pred2, lab, labl, loss = session.run([test_outputs, test_forced, label_indices, label_lengths, test_loss])
                        predictions.extend(assemble_string(p) for p in pred)
                        forced_predictions.extend(assemble_string(p) for p in np.argmax(pred2, -1))
                        label_strings.extend(assemble_string(l) for l in lab)
                        total_loss += np.sum(loss)
                        n_words = np.sum(labl)
                        total_words += n_words
                        pbar.update(n_words)
                    except tf.errors.OutOfRangeError:
                        break

            mean_loss = total_loss / total_words
            mean_perplexity = np.exp(mean_loss)
            bleu = compute_bleu([bleu_tokenize(ref) for ref in label_strings],
                                [bleu_tokenize(pred) for pred in predictions])

    with open(params.test_output, "w") as fh:
        fh.writelines(p + "\n" for p in predictions)

    with open(params.test_output + "2", "w") as fh:
        fh.writelines(p + "\n" for p in forced_predictions)

    with open("../verification.en", "w") as fh:
        fh.writelines(p + "\n" for p in label_strings)

    print("Loss: {:.3f}".format(mean_loss))
    print("Perplexity: {:.3f}".format(mean_perplexity))
    print("BLEU: {:.3f}".format(bleu))
