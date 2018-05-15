import tensorflow as tf

import ops


def rasor_net(context, context_len, question, question_len, span2position, embedding_matrix):
    # Passage-aligned question representations
    context_emb = ops.embed_sequence(inputs=context, name='embedding_layer', embedding_matrix=embedding_matrix)
    question_emb = ops.embed_sequence(inputs=question, name='embedding_layer', embedding_matrix=embedding_matrix)

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

    mask_pa = tf.cast(tf.not_equal(s_passage_aligned, 0.0), tf.float32)
    s_passage_aligned_exp = tf.exp(s_passage_aligned)
    s_passage_aligned_masked = mask_pa * s_passage_aligned_exp
    attn_passage_aligned = ops.normalize_linear(s_passage_aligned_masked)

    question_aligned = tf.matmul(attn_passage_aligned, question_dense)

    # Passage-independent question representation
    question_lstm_emb, _ = ops.bidirectional_lstm(
        inputs=question_emb,
        input_lengths=question_len,
        size=50,
        name='passage_independent_bLSTM'
    )

    s_passage_independent = tf.layers.dense(
        inputs=question_lstm_emb,
        units=1,
        activation=tf.nn.relu
    )

    mask_pi = tf.cast(tf.not_equal(s_passage_independent, 0.0), tf.float32)
    s_passage_independent_exp = tf.exp(s_passage_independent)
    s_passage_independent_masked = mask_pi * s_passage_independent_exp

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
        size=50,
        name='context_query_aware_lstm'
    )

    spans = []
    for k, v in span2position.items():
        # span_length = k[1] - k[0] + 1
        start = context_query_aware_lstm[:, k[0]]
        end = context_query_aware_lstm[:, k[1]]
        span_max = tf.reduce_max(context_query_aware_lstm[:, k[0]:k[1]+1], axis=1)
        span = tf.concat([start, end, span_max], axis=-1, name='span_{}'.format(v))
        spans.append(span)

    spans = tf.stack(spans, axis=1, name='stacked_span_representations')

    # Build a mask which masks out-of-bound spans
    # span_mask = tf.cast(tf.not_equal(tf.reduce_min(spans, axis=-1), 0.0), tf.float32)
    span_mask = tf.cast(tf.reduce_any(tf.not_equal(spans, 0), axis=-1), tf.float32)

    pre_logits = tf.layers.dense(
        inputs=spans,
        units=spans.get_shape()[-1].value,
        name='pre_final_ffn',
        activation=tf.nn.relu
    )

    prob = tf.placeholder_with_default(1.0, shape=(), name='out_dropout')

    logits_drouput = tf.nn.dropout(pre_logits, keep_prob=prob)

    logits = tf.layers.dense(
        inputs=logits_drouput,
        units=1,
        name='final_ffn',
    )

    logits = tf.squeeze(logits, axis=-1) * span_mask

    return logits, spans
