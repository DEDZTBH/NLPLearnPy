#!/usr/bin/env bash
bert-base-serving-start \
    -model_dir out_dir/ \
    -bert_model_dir init_checkpoint/ \
    -mode NER