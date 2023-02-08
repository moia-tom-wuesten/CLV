
.PHONY: lint
lint: lint-black lint-flake8

.PHONY: lint-black
lint-black:
	poetry run black --check src tests

.PHONY: lint-flake8
lint-flake8:
	poetry run flake8 --config=.flake8 src tests

.PHONY: fmt
fmt:
	poetry run black src tests

install:
	poetry install

test:
	poetry run pytest \
		-vv \
		--cov=src/CLV \
		--cov-report=html \
		--junitxml=test-results/junit.xml \
		tests
