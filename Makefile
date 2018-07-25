all:
	python Shadow.py --input_dir selected --mode train --output_dir output --max_epochs 10
	
test:
	python Shadow.py --input_dir selected --mode test --output_dir output --max_epochs 10 --checkpoint output

continue:
	python Shadow.py --input_dir selected --mode train --output_dir output --max_epochs 10 --checkpoint output