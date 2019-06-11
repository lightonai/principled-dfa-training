PRINCIPLED TRAINING OF NEURAL NETWORKS WITH DIRECT FEEDBACK ALIGNMENT
----------------------------

The `dfatools` and `mltools` folders are custom-made library to simplify classic ML tasks with or without DFA. 

`dfatools` contains our implementation of DFA.

`best_practices.py` contains code related to reproducing the tables for FC networks in section 3 and for CNNs in section 4. 

`bottleneck.py` contains the code for the bottlenecking experiments of section 4. 

`log_retrieval.py` and `bottleneck_from_log.py` are contingency code that were used for the bottlenecking experiments has we faced corruption issues with one of our files containing angle measurements.

When action is required to make the code work (such as adding paths to datasets), they are marked with a comment starting with "# TODO:".

Code author: [@slippylolo](https://github.com/slippylolo) (Julien Launay - julien[at]lighton.ai)