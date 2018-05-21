import tensorflow as tf

import ops


def rasor_net(context, context_len, question, question_len, span2position, embedding_matrix, span_mask):
    # Passage-aligned question representations
    context_emb = ops.embed_sequence(inputs=context, name='embedding_layer', embedding_matrix=embedding_matrix)
    question_emb = ops.embed_sequence(inputs=question, name='embedding_layer', embedding_matrix=embedding_matrix)

    context_mask = tf.cast(tf.expand_dims(tf.sequence_mask(context_len, context.get_shape()[1].value), -1), tf.float32)
    question_mask = tf.cast(tf.expand_dims(tf.sequence_mask(context_len, question.get_shape()[1].value), -1), tf.float32)

    context_dense = tf.layers.dense(
        inputs=context_emb,
        units=context_emb.get_shape()[-1].value,
        activation=tf.nn.relu
    )

    question_dense = tf.layers.dense(
        inputs=question_emb,
        units=question_emb.get_shape()[-1].value,
        activation=tf.nn.relu
    )

    s_passage_aligned = tf.matmul(context_dense, tf.transpose(question_dense, perm=[0, 2, 1]))

    s_passage_aligned_exp = tf.exp(s_passage_aligned)
    s_passage_aligned_masked = context_mask * s_passage_aligned_exp
    attn_passage_aligned = ops.normalize_linear(s_passage_aligned_masked)

    question_aligned = tf.matmul(attn_passage_aligned, question_dense)

    # Passage-independent question representation
    question_lstm_emb, _ = ops.bidirectional_lstm(
        inputs=question_emb,
        input_lengths=question_len,
        size=25,
        name='passage_independent_bLSTM'
    )

    s_passage_independent = tf.layers.dense(
        inputs=question_lstm_emb,    # span_mask = tf.cast(tf.not_equal(tf.reduce_min(spans, axis=-1), 0.0), tf.float32)
        units=1,
        activation=tf.nn.relu
    )

    s_passage_independent_exp = tf.exp(s_passage_independent)
    s_passage_independent_masked = question_mask * s_passage_independent_exp
    attn_passage_independent = ops.normalize_linear(s_passage_independent_masked)

    question_independent = attn_passage_independent * question_lstm_emb

    question_independent_final = tf.reduce_sum(question_independent, axis=1)
    question_independent_final_stacked = tf.stack([question_independent_final] * context.get_shape()[1].value, axis=1)

    # Concatenation
    context_query_aware = tf.concat([question_aligned, context_emb, question_independent_final_stacked], axis=-1)

    # Span representations
    context_query_aware_lstm, _ = ops.bidirectional_lstm(
        inputs=context_query_aware,
        input_lengths=context_len,
        size=25,
        name='context_query_aware_lstm'
    )

    spans = [None] * (max(span2position.values()) + 1)  # NOQA GET THE BLOODY THING WORKING
    for (start, end), position in span2position.items():
        # span_length = k[1] - k[0] + 1
        start = tf.gather(context_query_aware_lstm, [start], axis=1)
        end = tf.gather(context_query_aware_lstm, [end], axis=1)
        # span_max = tf.reduce_max(context_query_aware_lstm[:, k[0]:k[1]+1], axis=1)
        span = tf.squeeze(tf.concat([start, end], axis=-1, name='span_{}'.format(position)), axis=1)
        # spans.append(span)
        spans[position] = span

    assert not any([s is None for s in spans])

    spans = tf.stack(spans, axis=1, name='stacked_span_representations')

    # Build a mask which masks out-of-bound spans
    span_mask = tf.cast(tf.expand_dims(span_mask, -1), tf.float32)

    pre_logits = tf.layers.dense(
        inputs=spans * span_mask,
        units=spans.get_shape()[-1].value,
        name='pre_final_ffn',
        activation=tf.nn.relu
    )

    # prob = tf.placeholder_with_default(1.0, shape=(), name='out_dropout')
    # logits_dropout = tf.nn.dropout(pre_logits, keep_prob=prob)

    logits = tf.layers.dense(
        inputs=pre_logits*span_mask,
        units=1,
        name='final_ffn',
    )

    logits = tf.squeeze(logits * span_mask, axis=-1)
    return logits
