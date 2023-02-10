#!/bin/bash

# clas
python validation.py --input_dir_pd ../pd_models/paddleclas --output_dir_tlx ../tlx_models/paddleclas --model_name vgg16 --model_type clas
python validation.py --input_dir_pd ../pd_models/paddleclas --output_dir_tlx ../tlx_models/paddleclas --model_name alexnet --model_type clas

# nlp
python validation.py --input_dir_pd ../pd_models/paddlenlp --output_dir_tlx ../tlx_models/paddlenlp --model_name rnn --model_type nlp
python validation.py --input_dir_pd ../pd_models/paddlenlp --output_dir_tlx ../tlx_models/paddlenlp --model_name lstm --model_type nlp
python validation.py --input_dir_pd ../pd_models/paddlenlp --output_dir_tlx ../tlx_models/paddlenlp --model_name textcnn --model_type nlp
