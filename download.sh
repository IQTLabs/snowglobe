#!/usr/bin/env bash

mkdir -p models messages logs
wget https://code.jquery.com/jquery-3.7.1.min.js -P terminal
wget https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_M.gguf -P models
