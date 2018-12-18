import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nmt.model.layers import positional_encoding, _positional_encoding
import math


with tf.Graph().as_default():
    with tf.Session().as_default() as sess:
        encodings = positional_encoding(100, 20)
        encodings2 = _positional_encoding(100, 20)
        enc = sess.run(encodings)

# enc = np.zeros((100, 20))
# pos = np.arange(100, dtype=float)[:, np.newaxis]
# div_term = np.exp(np.arange(0, 20, 2, float) * -(math.log(10000) / 20))
# enc[:, ::2] = np.sin(pos * div_term)
# enc[:, 1::2] = np.cos(pos * div_term)
#
# np.testing.assert_allclose(enc, enc2)

plt.figure(figsize=(15, 5))
plt.plot(np.arange(enc.shape[0]), enc[:, 4:8])
# plt.plot(np.arange(encodings.shape[0]), encodings[:, 260:262])
plt.legend(["dim %d"%p for p in [4,5,6,7]])
plt.show()
