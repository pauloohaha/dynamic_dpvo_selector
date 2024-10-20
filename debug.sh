#CUDA_VISIBLE_DEVICES=2 python -m debugpy --listen 10.0.1.231:3493 --wait-for-client test_model.py
CUDA_VISIBLE_DEVICES=2 python -m debugpy --listen 10.0.1.231:3493 --wait-for-client train.py