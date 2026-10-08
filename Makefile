.PHONY: install data cap test
install:
	pip install -e ".[dev]"
data:
	python -m fuellab.pipeline aggregate-fr --raw data/raw/PrixCarburants_annuel_2026.xml
cap:
	python -m fuellab.pipeline cap
test:
	pytest -q
