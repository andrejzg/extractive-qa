import tensorflow as tf

import ops


def rasor_net(context, context_len, question, question_len, span2position, embedding_matrix, span_mask, is_training):
    # Passage-aligned question representations
    context_emb = ops.embed_sequence(inputs=context, name='embedding_layer', embedding_matrix=embedding_matrix)
    question_emb = ops.embed_sequence(inputs=question, name='embedding_layer', embedding_matrix=embedding_matrix)

    context_mask = tf.cast(tf.expand_dims(tf.sequence_mask(context_len, context.get_shape()[1].value), -1), tf.float32)
    question_mask = tf.cast(tf.expand_dims(tf.sequence_mask(context_len, question.get_shape()[1].value), -1), tf.float32)

    cq_ff = tf.layers.Dense(
        units=context_emb.get_shape()[-1].value,
        activation=tf.nn.relu,
        use_bias=True,
        name='cq_ff'
    )

    context_dense = cq_ff(context_emb)
    context_dense = tf.layers.dropout(
        inputs=context_dense,
        training=is_training,
        rate=0.2,
        name='context_dense_dropout'
    )

    question_dense = cq_ff(question_emb)
    question_dense = tf.layers.dropout(
        inputs=question_dense,
        training=is_training,
        rate=0.2,
        name='question_dense_dropout'
    )

    s_passage_aligned = tf.matmul(context_dense, tf.transpose(question_dense, perm=[0, 2, 1]))

    attn_passage_aligned = ops.masked_softmax(s_passage_aligned, context_mask)

    question_aligned = tf.matmul(attn_passage_aligned, question_dense)

    # Passage-independent question representation
    question_lstm_emb, _ = ops.bidirectional_lstm(
        inputs=question_emb,
        input_lengths=question_len,
        size=100,
        name='passage_independent_bLSTM'
    )

    s_passage_independent = tf.layers.dense(
        inputs=question_lstm_emb,    # span_mask = tf.cast(tf.not_equal(tf.reduce_min(spans, axis=-1), 0.0), tf.float32)
        units=1,
        activation=tf.nn.relu
    )
    s_passage_independent = tf.layers.dropout(
        inputs=s_passage_independent,
        training=is_training,
        rate=0.2,
        name='s_passage_independent_dropout'
    )

    attn_passage_independent = ops.masked_softmax(s_passage_independent, question_mask)

    question_independent = attn_passage_independent * question_lstm_emb

    question_independent_final = tf.reduce_sum(question_independent, axis=1)
    question_independent_final_stacked = tf.stack([question_independent_final] * context.get_shape()[1].value, axis=1)

    # Concatenation
    context_query_aware = tf.concat([question_aligned, context_emb, question_independent_final_stacked], axis=-1)

    # Span representations
    context_query_aware_lstm, _ = ops.bidirectional_lstm(
        inputs=context_query_aware,
        input_lengths=context_len,
        size=100,
        name='context_query_aware_lstm'
    )

    span_start_dense = tf.layers.dense(
        inputs=context_query_aware_lstm,
        units=context_query_aware_lstm.get_shape()[-1].value,
        name='end_span_ff',
        use_bias=False
    )

    span_start_dense = tf.layers.dropout(
        inputs=span_start_dense,
        training=is_training,
        rate=0.2,
        name='span_start_dense_dropout'
    )

    span_end_dense = tf.layers.dense(
        inputs=context_query_aware_lstm,
        units=context_query_aware_lstm.get_shape()[-1].value,
        name='start_span_ff',
        use_bias=False
    )

    span_end_dense = tf.layers.dropout(
        inputs=span_end_dense,
        training=is_training,
        rate=0.2,
        name='span_start_dense_dropout'
    )

    spans = [None] * len(span2position)  # NOQA GET THE BLOODY THING WORKING
    for (start, end), position in span2position.items():
        # span_length = k[1] - k[0] + 1
        start = tf.gather(span_start_dense, [start], axis=1)
        end = tf.gather(span_end_dense, [end], axis=1)
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
        activation=tf.nn.relu,
    )

    # prob = tf.placeholder_with_default(1.0, shape=(), name='out_dropout')
    # logits_dropout = tf.nn.dropout(pre_logits, keep_prob=prob)

    logits = tf.layers.dense(
        inputs=pre_logits,
        units=1,
        name='final_ffn',
        use_bias=False
    )

    logits = tf.squeeze(logits * span_mask, axis=-1)

    return logits, context_query_aware_lstm, context_query_aware
