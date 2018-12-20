from argparse import ArgumentParser
import tensorflow as tf
from nmt import train, test
from preprocess import preprocess


def main(_):
    if params.test:
        test(params)
    elif params.preprocess:
        preprocess(params)
    else:
        train(params)


parser = ArgumentParser()
parser.add_argument("--max_gpus", type=int, default=99)
parser.add_argument("--source_vocabulary_file", type=str, default="../WMT2014/training/en.vocabulary")
parser.add_argument("--target_vocabulary_file", type=str, default="../WMT2014/training/de.vocabulary")
# parser.add_argument("--vocabulary_file", type=str, default="../WMT2014/bilingual.vocabulary")
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=3000)  # paper: 25000
parser.add_argument("--model_dim", type=int, default=512)
parser.add_argument("--warmup_steps", type=int, default=4000)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.98)
parser.add_argument("--adam_epsilon", type=float, default=1e-9)
parser.add_argument("--train_dir", type=str, default="../train")
parser.add_argument("--event_dir", type=str, default="../event")
parser.add_argument("--output_dir", type=str, default="../output")
parser.add_argument("--training_source", type=str, default="../WMT2014/training/europarl-v7.de-en.en.bpe")
parser.add_argument("--training_target", type=str, default="../WMT2014/training/europarl-v7.de-en.de.bpe")
parser.add_argument("--development_source", type=str, default="../WMT2014/dev/newstest2013.en.bpe")
parser.add_argument("--development_target", type=str, default="../WMT2014/dev/newstest2013.de.bpe")
parser.add_argument("--development_samples", type=int, default=81355)
parser.add_argument("--test_source", type=str, default="../WMT2014/test-full/newstest2014-deen-src.en.bpe")
parser.add_argument("--test_target", type=str, default="../WMT2014/test-full/newstest2014-deen-src.de.bpe")
parser.add_argument("--test_samples", type=int, default=86821)
parser.add_argument("--model_name", type=str, default="NMT")
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=2000)
parser.add_argument("--beam_size", type=int, default=4)
parser.add_argument("--length_penalty", type=float, default=0.6)
parser.add_argument("--training_steps", type=int, default=100000)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--hidden_size", type=int, default=2048)
parser.add_argument("--num_encoder_blocks", type=int, default=6)
parser.add_argument("--num_decoder_blocks", type=int, default=6)
parser.add_argument("--max_example_length", type=int, default=100)
parser.add_argument("--label_smoothing", type=int, default=0.1)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--accumulate_gradients", type=int, default=8)
parser.add_argument("--test", action="store_true")
parser.add_argument("--preprocess", action="store_true")

params = parser.parse_args()

if __name__ == "__main__":
    tf.app.run()
