# Positive negative review classifier

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

Go to folder with project -> Click on the Address Bar -> Type "cmd"

```bash
pip install -r requirements.txt
```

## Usage

```bash
python inference.py test_reviews.csv --output_file the_predictions.csv
```

## Input
test_reviews.csv - Path to the input CSV file (e.g., test_reviews.csv)

test_reviews.csv has two columns: 'id' (id of review) and 'text' (text of review)

## Output
the_predictions.csv - Path to the output CSV file with predictions

the_predictions.csv has two columns: 'id' (id of review) and 'sentiment' (Positive, Negative)
