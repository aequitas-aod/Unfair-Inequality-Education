# Unfair Inequality in Education: A Benchmark for AI-Fairness Research

## Abstract

This repository proposes a novel benchmark specifically designed for AI fairness research in education.
It can be used for challenging tasks aimed at improving students' performance and reducing dropout rates which are also discussed in the paper to emphasize significant research directions.
By prioritizing fairness,  this benchmark aims to foster the development of bias-free AI solutions, promoting equal educational access and outcomes for all students.

## Structure

```benchmark```
contains:
- the proposed dataset (```dataset.csv```),
- the mask for dealing with missing values (```missing_mask.csv```), and
- the meta-columns providing grouping criteria and sample weights for each student (```meta_cols.csv```).

```raw_data``` includes:
- the original dataset (```original.csv```), and
- the intermediate stages of the pre-processing and validation pipeline (```split```, ```pre_processed```, ```validation```).

```res``` contains the documentation, including:
- the transformation mapping each column of the original dataset to the proposed one, along with missingness category and original text (```meta_data_mapping.csv```),
- the value type and domains of each column of the proposed datasets (```meta_data_stats.json```), and
- the statistical indices of the validation pipeline (```bias_preservation_results.json```).

```src``` contains the source code for running the pre-processing and corresponding analysis:
- ```pre_processing``` and ```stats```contain the code for the two corresponding tasks, and
- ```pre_processing.py``` and ```split.py``` are two entry points.

Finally, ```Dockerfile``` and ```requirements.txt``` set up the environment for running the applications across multiple platforms and with Python, respectively.
