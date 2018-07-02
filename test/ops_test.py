import numpy as np
import tensorflow as tf

import ops
import data_ops


class BasicOpsTest(tf.test.TestCase):
    def test_normalize_linear(self):
        """
        Tests normalize_linear which does divides each row of M
        by its sum, thus making each row add up to 1.
        """
        with self.test_session():
            M = [[[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
                 ]  # nan edge case (will lead to division by 0)

            M = np.array(M)
            M = tf.constant(M)

            N = ops.normalize_linear(tf.cast(M, dtype=tf.float32))
            n = tf.reduce_sum(N, axis=-1).eval()

            np.testing.assert_almost_equal(n, np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]))

    def test_pad_mask(self):
        with self.test_session():
            M = [[[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
                 ]  # nan edge case (will lead to division by 0)

            M = np.array(M)
            K = tf.constant(M, tf.float32)

            mask_pi = tf.cast(tf.not_equal(K, 0.0), tf.float32)

            res = mask_pi * K
            np.testing.assert_almost_equal(res.eval(), M)

    def test_new_pad_mask_creation(self):
        with self.test_session():
            M = [[[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
                 ]

            M = np.array(M)
            K = tf.constant(M, tf.float32)

            s_lens = [4, 3, 3]
            s_lens = np.array(s_lens)
            s_lens = tf.constant(s_lens, tf.int32)

            mask = tf.cast(tf.expand_dims(tf.sequence_mask(s_lens, 6), -1), tf.float32)

            res = mask * K
            np.testing.assert_almost_equal(res.eval(), M)

    def test_span_mask(self):
        with self.test_session():
            M = [[[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 1, 1, 0],
                  [5, 5, 0, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
                 ]

            M = np.array(M)
            K = tf.constant(M, tf.float32)

            s2p = data_ops.make_span2position(seq_size=6, max_len=4)

            mask_values = np.asarray(list({k: v for k, v in s2p.items() if np.all(np.asarray(k) <= 4)}.values()))
            mask = np.zeros(len(s2p.items()))
            mask[mask_values] = 1
            mask = tf.constant(mask, tf.float32)
            mask = tf.stack([mask] * 3)
            spans = [None] * (max(s2p.values()) + 1)

            for (start, end), position in s2p.items():
                # span_length = k[1] - k[0] + 1
                start = tf.gather(K, [start], axis=1)
                end = tf.gather(K, [end], axis=1)
                # span_max = tf.reduce_max(context_query_aware_lstm[:, k[0]:k[1]+1], axis=1)
                span = tf.squeeze(tf.concat([start, end], axis=-1, name='span_{}'.format(position)), axis=1)
                # spans.append(span)
                spans[position] = span

            spans = tf.stack(spans, axis=1, name='stacked_span_representations')
            masked_spans = spans * tf.expand_dims(mask, -1)

            for i in list(np.where(mask.eval()[0] == 0)[0]):
                assert sum(masked_spans[0][i].eval()) == 0.0

    def test_masked_softmax1(self):
        with self.test_session():
            M = [[5.0, 1.0, -3.0, 2.0],
                 [1.0, 1.0, 2.0, 9.0],
                 [0.5, -0.1, 100.0, 5.0],
                 [-2.0, -90.0, -2.0, -1.0]]

            idx = tf.cast(tf.reshape(tf.argmax(M, axis=1), [-1, 1]), tf.int32)
            rows = tf.reshape(tf.range(idx.get_shape()[0]), [-1, 1])

            capture = tf.concat([rows, idx], axis=1)

            res = tf.gather_nd(M, capture)

            np.testing.assert_almost_equal(res.eval(), [5.0, 9.0, 100.0, -1.0])

    def test_masked_softmax2(self):
        with self.test_session():
            M = [[5.0, 1.0, -3.0, 2.0],
                 [1.0, 1.0, 2.0, 9.0],
                 [0.5, -0.1, 100.0, 5.0],
                 [-2.0, -90.0, -2.0, -1.0]]

            idx = tf.argmax(M, axis=1)
            mask = tf.one_hot(idx, depth=idx.get_shape()[-1], dtype=tf.bool, on_value=True, off_value=False)
            res = tf.boolean_mask(M, mask)

            np.testing.assert_almost_equal(res.eval(), [5.0, 9.0, 100.0, -1.0])



