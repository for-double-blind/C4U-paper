# C4U-paper
This is the repository for the code and data of paper: Cross-domain-aware Worker Selection with Training for Crowdsourced Annotation

## Settings
Intel Xeon Gold 6240 CPU @ 2.60GHz

Python version 3.8

## Install
```console
$ git clone [link to repo]
$ cd C4U-paper
$ pip install -r requirements.txt 
```

If you are using Anaconda, you can create a virtual environment and install all the packages:

```console
$ conda create --name C4U python=3.8
$ conda activate C4U
$ pip install -r requirements.txt
```

## Reproduce the results
In order to reproduce the results on the real-world dataset, please run the following code:
```console
$ python real-exp.py >> ./real_result.txt
```

In order to reproduce the results on the synthetic datasets, please run the following code:
```console
$ python synthetic-exp.py >> ./synthetic_result.txt
```

You can change the value of the corresponding parameters in synthetic-exp.py to achieve the results in different synthetic datasets in our paper.
