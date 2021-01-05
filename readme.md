This repo contains the code for digits experiments SVHN->MNIST of the Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation
 paper.

For installation instructions, please refer to [implicit_alignment](https://github.com/xiangdal/implicit_alignment).

The digits data are defined in [digit.py](./ai/domain_adaptation/datasets/digits.py).
By default, this repo runs experiments for RS-UT where both source and target domains are unbalanced under mild unbalance.
This configuration can be modified in [digit.py](./ai/domain_adaptation/datasets/digits.py).

Running this repo with [makefile](./ai/domain_adaptation/makefiles/Makefile_digits) should yield the following results (with variations of random seeds and pytorch versions):

- source-only: 61%
- DANN: 67%
- DANN+implicit: 88%

DANN is implemented based on modifications of the MDD code.

Note, this repo only supports SVHN->MNIST, for other datasets please refer to the main repo [implicit_alignment](https://github.com/xiangdal/implicit_alignment).

Please create an issue for question/bug reports. Thanks!