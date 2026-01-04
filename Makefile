.PHONY: setup data train run test docker-build docker-run

setup:
	python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

data:
	python src/download_data.py

train:
	python src/train.py

run:
	python src/predict.py

test:
	pytest -q

docker-build:
	docker build -t energy-grid-stress .

docker-run:
	docker run -p 9696:9696 energy-grid-stress
