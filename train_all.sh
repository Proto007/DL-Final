#!/bin/bash

# 'sample', 'se', 'hcnn'
for arg in "$@"; do
  if [[ "$arg" == "fcn" ]]; then
    python main.py --model_type fcn --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type fcn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type fcn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type fcn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type fcn --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "musicnn" ]]; then
    python main.py --model_type musicnn --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type musicnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type musicnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type musicnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type musicnn --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "crnn" ]]; then
    python main.py --model_type crnn --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type crnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type crnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type crnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type crnn --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "sample" ]]; then
    python main.py --model_type sample --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type sample --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type sample --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type sample --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type sample --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "se" ]]; then
    python main.py --model_type se --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type se --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type se --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type se --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type se --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "short" ]]; then
    python main.py --model_type short --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type short --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type short --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type short --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type short --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "short_res" ]]; then
    python main.py --model_type short_res --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type short_res --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type short_res --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type short_res --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type short_res --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  elif [[ "$arg" == "hcnn" ]]; then
    python main.py --model_type hcnn --num_workers 8 --n_epochs 10 --no-aug
    python main.py --model_type hcnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.25
    python main.py --model_type hcnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.50
    python main.py --model_type hcnn --num_workers 8 --n_epochs 10 --aug --aug_prob 0.75
    python main.py --model_type hcnn --num_workers 8 --n_epochs 10 --aug --aug_prob 1.00
  fi
done
exit 0

