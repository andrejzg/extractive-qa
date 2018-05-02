import numpy as np
import tensorflow as tf

import ops


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

