python3 -m venv project
DIR="project/bin"

if [ -d "$DIR" ]; then
	source project/bin/activate
	pip install -r requirements.txt
	pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
	pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
	pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
fi
