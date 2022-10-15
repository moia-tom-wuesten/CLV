
format:
	poetry run black --line-length=100 .

format-check:
	poetry run flake8 --config=.flake8

lint: format format-check

install:
	poetry install

test:
	poetry run coverage run --source=voice-assistant-restapi -m pytest -v -p no:warnings
	poetry run coverage report -m --skip-covered

docker:
	docker build --tag=voice-assistant-restapi --memory=2g .
