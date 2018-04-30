import numpy as np
import tensorflow as tf

import ops


class BasicOpsTest(tf.test.TestCase):
    def test_cosine_sim(self):
        with self.test_session():
            A = tf.constant(np.array([[
                [1, 1, 1, 1, 0],
                [1, 2, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]]), dtype=tf.float32)

            row_len = 4
            col_len = 4

            ans = A.eval()
            ans[0, :row_len, :col_len] = np.divide(np.exp(ans[0, :row_len, :col_len]), np.sum(ans[0, :row_len, :col_len], axis=-1))
            #
            # mask = np.zeros_like(A.eval())
            # mask[0, :row_len, :col_len] = 1
            # mask = tf.constant(mask)
            #
            # A_exp = mask * tf.exp(A)
            # res = ops.normalize_linear(A_exp)
            #
            # import code
            # code.interact(local=locals())
            #
            # np.testing.assert_almost_equal(
            #     ans,
            #     res
            # )

