.PHONY: build

build:
	python ./scripts/build.py --pre
	poetry build
	python ./scripts/build.py --post

.PHONY: all
all: build
