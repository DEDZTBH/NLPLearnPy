#!/usr/bin/env bash
bert-base-serving-start \
    -model_dir out_dir/ \
    -bert_model_dir /Users/peiqi/PycharmProjects/NLPLearn/bert-bilstm-crf-ner/work/init_checkpoint/ \
    -mode NER