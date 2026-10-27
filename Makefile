.PHONY: install test run pdf app all clean

install:
	pip install -r requirements.txt

test:
	pytest fhe_brain_tumor/tests/ -v

run:
	python -m fhe_brain_tumor.main

pdf:
	python -m fhe_brain_tumor.generate_summary_pdf

app:
	streamlit run app.py

all: test run pdf

clean:
	python -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['__pycache__', '.pytest_cache', 'fhe_brain_tumor/__pycache__', 'fhe_brain_tumor/tests/__pycache__']]"
