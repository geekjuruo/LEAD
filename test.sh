#!/usr/bin/env bash

name="test13"
output_dir=output/$name
checkpoint=$name.pkl

mkdir -p $output_dir

python -u main.py --config config/$name.yaml --checkpoint $checkpoint --variables meta.mode=test,output.directory=$output_dir