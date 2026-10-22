# Grape Phenotyping

Sample data and core measurement code accompanying “Phenotypic parameter extraction of postharvest table grapes based on point cloud data.”

The program calculates grape bunch height, width, thickness, surface area, volume, and berry count from a preprocessed three-dimensional point cloud. The supplied dataset corresponds to Sample 1 and allows reviewers to run the measurement workflow directly.

## Repository contents

- `grape_phenotyping.py`: standalone Python implementation of the measurement algorithms.
- `data.ply`: the preprocessed Sample 1 point cloud for test.
- `README.md`: installation and execution instructions.

## Requirements

The code has been tested with:

- Python 3.11.15
- NumPy 2.4.4
- SciPy 1.17.1

In a Python 3.11 environment, install the dependencies using:

```bash
python -m pip install numpy==2.4.4 scipy==1.17.1
```

No pretrained model weights or additional project scripts are required.

## Running the example

Download the repository and keep `grape_phenotyping.py` and `data.ply` in the same directory. Open a terminal in that directory and run:

```bash
python grape_phenotyping.py
```

## Input data

The supplied `data.ply` contains 74,863 valid points. Coordinates are expressed in **metres**. The input must include XYZ coordinates; RGB colours are optional.The point cloud has already been preprocessed and separated from the reference ball. 

## Output

The program generates **`phenotype_results.csv`**, directly in the directory containing `grape_phenotyping.py`. Running the program again overwrites this CSV file.The CSV contains computed predictions. Full numerical precision is retained in the CSV. 

