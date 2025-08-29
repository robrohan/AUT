
all: build_roberta build_biobert build_clinicalbert

build_roberta:
	docker build -f Dockerfile.distilroberta -t robrohan/transformers-inference:distilroberta-inference .

build_clinicalbert:
	docker build -f Dockerfile.clinicalbert -t robrohan/transformers-inference:clinicalbert-inference .

build_biobert:
	docker build -f Dockerfile.biobert-base-cased -t robrohan/transformers-inference:biobert-base-cased .
