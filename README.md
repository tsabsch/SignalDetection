# SignalDetection

This repository contains an analysis of the [HEPMASS](archive.ics.uci.edu/ml/datasets/HEPMASS) data set from the _UCI Machine Learning Repository_. It's goal is to detect signals generated by particle-producing collisions. The analysis consists of the following parts:

* Description of the data set and its attributes
* Data preprocessing (e.g. Cleaning, handling missing values, ...)
* Classification by several classification strategies
* Evaluation of the results

## Installation

The analysis is performed with python3 and visualised in [Jupyter Notebooks](http://jupyter.org/). You can install and run jupyter with the following commands:

```
$ pip install jupyter
$ jupyter notebook
```

We use [dask dataframes](http://dask.pydata.org/en/latest/) to read and process large csv files. You can install it by running

```
$ pip install dask[dataframe]
```

## Usage

This analysis relies on large data files, which will not be uploaded to Github. Code blocks which use these data files will not properly be displayed online. To run these code blocks, download the repository via HTTPS or SSH:
```
$ git clone https://github.com/tsabsch/SignalDetection.git
$ git clone git@github.com:tsabsch/SignalDetection.git
```
and run it on a local jupyter instance.

The data files are assumed to be within the directory `data`. If you have placed your data files elsewhere, you have to update the code to fit your path.

## Remarks
This analysis has been made as part of the course [__Advanced Topics in Machine Learning__](http://www.findke.ovgu.de/findke/en/Studies/Courses/Summer+Term+2017/Advanced+Topics+in+Machine+Learning.html) at the Otto-von-Guericke-University Magdeburg.
