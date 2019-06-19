#!/usr/bin/env bash
bert-base-ner-train \
    -data_dir data_dir/ \
    -output_dir out_dir/ \
    -init_checkpoint init_checkpoint/bert_model.ckpt \
    -bert_config_file init_checkpoint/bert_config.json \
    -vocab_file init_checkpoint/vocab.txt