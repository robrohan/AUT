
install:
	pip install -r requirements.txt
	brew install timidity

test_to_wav:
	mkdir -p output
	timidity ./output/batch.mid -Ow -o ./output/batch.wav
	timidity ./output/test.mid -Ow -o ./output/test.wav
