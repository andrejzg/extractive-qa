import tensorflow as tf

import ops


def rasor_net(context, context_len, question, question_len, span2position):
    # Passage-aligned question representations
    context_emb = ops.embed_sequence(inputs=context, name='embedding_layer')
    question_emb = ops.embed_sequence(inputs=question, name='embedding_layer')

    context_dense = tf.layers.dense(
        inputs=context_emb,
        units=context_emb.get_shape()[-1].value
    )

    question_dense = tf.layers.dense(
        inputs=question_emb,
        units=question_emb.get_shape()[-1].value,
    )

    s_passage_aligned = tf.matmul(context_dense, tf.transpose(question_dense, perm=[0, 2, 1]))

    mask_pa = tf.cast(tf.equal(s_passage_aligned, 0.0), tf.float32)
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
    )

    mask_pi = tf.cast(tf.equal(s_passage_independent, 0.0), tf.float32)
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
        start = context_query_aware_lstm[:, k[0]]
        end = context_query_aware_lstm[:, k[1]]
        span = tf.concat([start, end], axis=-1, name='span_{}'.format(v))
        spans.append(span)

    spans = tf.stack(spans, axis=1, name='stacked_span_representations')

    logits = tf.layers.dense(
        inputs=spans,
        units=1,
        name='final_dense_layer'
    )

    return tf.squeeze(logits, axis=-1)
