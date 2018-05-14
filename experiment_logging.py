"""
Author: Sam Coope
"""

import tensorflow as tf
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


def print_spans(
        to_pick,
        context_ids,
        question_ids,
        predicted_labels,
        span_color,
        position2span,
        id2word,
        ignore_ids=(0, 1)
        ):
    contexts = [' '.join([id2word[w] if w not in ignore_ids else '' for w in x]).strip() for x in context_ids[to_pick]]
    questions = [' '.join([id2word[w] if w not in ignore_ids else '' for w in x]).strip() for x in question_ids[to_pick]]

    span_positions = [np.where(x == 1)[0].tolist() for x in predicted_labels[to_pick]]

    for context, question, span_position in zip(contexts, questions, span_positions):
        to_print_context = []
        span_starts = [position2span[p][0] for p in span_position]
        span_ends = [position2span[p][1] for p in span_position]
        for i, w in enumerate(context.split()):
            if i in span_starts:
                to_print_context.append(span_color)
            to_print_context.append(w)
            if i in span_ends:
                to_print_context.append('\x1b[0m')
        print(' '.join(to_print_context))
        print(f'Q: {question}')
        print('\n')


def select_n_classified(
        ground_truth,
        predicted,
        correct=True, n=5
        ):
    # compare ground-truth labels with predicted labels
    elem_comp = np.equal(predicted, ground_truth)

    # Find rows which match exactly to find correctly predicted examples
    correct_rows = np.all(elem_comp, axis=1)

    # Depending on the correct=True/False flag either pick n correct examples or n incorrectly classified ones
    selected_row_idx = (correct_rows == correct).nonzero()[0]
    if len(selected_row_idx) > 0:
        # set() used in case n > len(selected_row_idx)
        to_pick = set(np.random.choice(selected_row_idx, n, replace=True))
        to_pick = list(to_pick)
        return to_pick
