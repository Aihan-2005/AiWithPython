.PHONY: install train lr-find tune-random tune-hyperband tune-bayesian test tensorboard

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

train:
	fashion-mlp-train --model sequential --hidden-units 300 100 --epochs 50

lr-find:
	fashion-mlp-lr-find --start-lr 1e-5 --end-lr 1 --steps 500

tune-random:
	fashion-mlp-tune --strategy random --max-trials 12 --epochs 30

tune-hyperband:
	fashion-mlp-tune --strategy hyperband --epochs 30

tune-bayesian:
	fashion-mlp-tune --strategy bayesian --max-trials 15 --epochs 30

test:
	pytest -q

tensorboard:
	tensorboard --logdir runs
