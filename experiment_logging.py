"""
Author: Sam Coope
"""

import tensorflow as tf
from io import StringIO
import numpy as np


class TensorboardLogger(object):
    """Logging in TensorBoard without TensorFlow ops."""

    def __init__(self, log_dir=None, writer=None):
        if writer is None:
            assert log_dir is not None, 'Must provide a logdir or a summary writer'
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = writer

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        

    def log_histogram(self, tag, values, step, bins=1000, min=None, max=None, density=False):
        """Logs the histogram of a list/vector of values."""
        values = np.array(values)
        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values)) if min is None else float(min)
        hist.max = float(np.max(values)) if min is None else float(max)

        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins, range=(hist.min, hist.max), density=density)

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()