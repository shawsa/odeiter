.PHONY: test badge

test:
	uv run pytest

badge:
	uv run pytest --junitxml=reports/junit/junit.xml
	uv run genbadge tests -o badges/tests.svg
	uv run coverage report
	uv run coverage xml -o reports/coverage/coverage.xml
	uv run genbadge coverage -o badges/coverage.svg
