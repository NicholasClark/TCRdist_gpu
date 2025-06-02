# A Fast GPU-compatible Implementation of TCRdist

## Overview

**TCRdist** is a method for quantifying the **similarity of two T-cell receptors (TCRs)** based on the V-segments and CDR3 regions of their alpha and beta chains ([Dash et al., Nature 2017](https://doi.org/10.1038/nature22383)). It is a **similarity-weighted Hamming distance**, with a higher weight given to the CDR3 sequence in recognition of its disproportionate role in epitope specificity and a gap penalty introduced to capture variation in length. The mismatch penalties used by TCRdist are based on the [BLOSUM62 substitution matrix](https://en.wikipedia.org/wiki/BLOSUM).

The `TCRdist_batch()` function is a **GPU-compatible Python re-implementation of TCRdist** that can efficiently calculate pairwise similarities of thousands or millions of TCRs. In testing, we were able to **calculate pairwise TCRdist** for a repertoire of **~1 million TCRs in a few hours** using a MacBook Pro (16-core GPU, M4 Pro). It **works with both NVIDIA GPUs**, using [cupy](https://cupy.dev/), **and Apple Silicon GPUs**, using [mlx](https://opensource.apple.com/projects/mlx/).

In order to **facilitate analysis of large TCR datasets**, the function **calculates pairwise TCRdist in batches** (default batch size is 1000 TCRs) and **returns sparse output**, only including pairs of TCRs with TCRdist less than a user-specified cutoff (default cutoff is TCRdist <= 90).

The **[TIRTLtools R package](https://github.com/NicholasClark/TIRTLtools)**, a package written to provide analysis and visualization functions for paired-chain [TIRTL-seq](https://github.com/pogorely/TIRTL) data, contains an **R wrapper function** `TCRdist()` that calls `TCRdist_batch()` through the `reticulate` package.

## Links and References

For details on TCRdist, see [Dash et al., Nature 2017](https://doi.org/10.1038/nature22383)

For details on the TIRTLseq TCR pairing pipeline, see the [TIRTLseq preprint (Pogorelyy and Kirk et al.)](https://www.biorxiv.org/content/10.1101/2024.09.16.613345v2) and our [github repository](https://github.com/pogorely/TIRTL).

<br>
Written by Mikhail Pogorelyy and Nicholas Clark, [Thomas Lab, St. Jude Children's Research Hospital](https://www.stjude.org/research/labs/thomas-lab.html)