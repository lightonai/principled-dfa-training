# Principled Training of Neural Networks with Direct Feedback Alignment

This is the code for reproducing the results of our paper:

**Principled Training of Neural Networks with Direct Feedback Alignment**
*Julien Launay, Iacopo Poli, Florent Krzakala*



## Abstract

The backpropagation algorithm has long been the canonical training method for neural networks. Modern paradigms are implicitly optimized for it, and numerous guidelines exist to ensure its proper use. Recently, synthetic gradients methods -- where the error gradient is only roughly approximated -- have garnered interest. These methods not only better portray how biological brains are learning, but also open new computational possibilities, such as updating layers asynchronously. Even so, they have failed to scale past simple tasks like MNIST or CIFAR-10. This is in part due to a lack of standards, leading to ill-suited models and practices forbidding such methods from performing to the best of their abilities. In this work, we focus on direct feedback alignment and present a set of best practices justified by observations of the alignment angles. We characterize a bottleneck effect that prevents alignment in narrow layers, and hypothesize it may explain why feedback alignment methods have yet to scale to large convolutional networks.



## Reproducing the results

Running the provided code requires a CUDA-enabled GPU with around 1GB of memory. 

When actions are required to make the code work (such as adding paths to datasets), they are marked with a comment starting with `# TODO:”`. 

To reproduce, 

- `best_practices.py` contains code related to the tables for FC networks in section 3 and for CNNs in section 4;
- `bottleneck.py` contains the code for the bottlenecking experiments of section 4, and `log_retrieval.py` and `bottleneck_from_log.py` are contingency codes that were used as a file containing angle measurements value was corrupted.

Furthermore, the  `mltools` folder is a custom-made library to simplify classic ML tasks with or without DFA. `dfatools` contains our implementation of DFA. A more streamlined and complete version of it will be released eventually.

## Citation

If you found this implementation useful in your research, please consider citing:

```
@inproceedings{launay2019principled,
    title={Principled Training of Neural Networks with Direct Feedback Alignment},
    author={Launay, Julien and Poli, Iacopo and Krzakala, Florent},
    booktitle={Preprint},
    year={2019}
}
```

Code author: [@slippylolo](https://github.com/slippylolo) (Julien Launay - julien[at]lighton.ai)