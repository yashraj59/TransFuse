# TransFuse: Transfer learning on multi-omic graph fusion network

TransFuse v1.0
Yan's Lab, Indiana University Indianapolis
Developed by Linhui Xie

## Description
TransFuse is an interpretable multi-omic graph neural network approach for integratively identifying network-level biomarkers of target diseases. Originally forked from Varmole(https://github.com/namtk/Varmole). TransFuse addresses the incompleteness issue of multi-omic data through a transfer learning approach. By pretraining on single-omic connections and transferring the learned knowledge to multi-omic connections, TransFuse ensures more accurate integration and interpretation of diverse biological data types. It incorporates individual connections between protein-protein, SNP-gene, and gene-protein interactions, providing a more detailed understanding of biological networks compared to a single transparent layer integrating quantitative trait loci (QTLs) and gene regulatory networks (GRNs) into prior biological knowledge.

TransFuse extends its capabilities by modeling several critical biological interactions: genotypic variations (e.g., SNPs) affecting gene expression, gene-protein regulatory networks, and protein-protein interaction networks. Additionally, it integrates gene expression and protein levels to model proteoforms or protein complexes, offering a comprehensive view of the biological systems. This multi-omic graph neural network not only enhances the identification of biomarkers but also aids in understanding the underlying mechanisms of target diseases through its robust and interpretable framework.


## Installation
The following libraries are required,
Python >= 3.7
pandas >= 1.2
numpy >= 1.20 
PyTorch >= 0.5
captum >= 0.3.0
argparse >= 0.1.0
sklearn >= 1.0.2


## Usage
Prepare your multi-omic matrix data in .csv format. Each row should contain the multi-omic data for a single participant, with columns representing the multi-omic features. The adjacency matrix should also be in .csv format, with rows and columns corresponding to the total number of multi-omic features. Place all input files in the same directory.

To run the code, use one of the following commands in your terminal, specifying the path to your dataset folder and the desired training mode:
$ python transfuse_main.py /the_path_to_your_dataset/ pretrain
$ python transfuse_main.py /the_path_to_your_dataset/ baseline
$ python transfuse_main.py /the_path_to_your_dataset/ transferweight
$ python transfuse_main.py /the_path_to_your_dataset/ fine-tune


## Contents
You can request access the real dataset through AD knowledge portal(https://adknowledgeportal.synapse.org) with synapse #.
For the detailed information under this repository, the root folder contains following scripts,
transfuse_main.py contains the main structure.
model.py contains all input information to the pipeline .
train.py contains the training part of this method.
utils.py contains all utility functions that will be applied in the approach.


## Information
 * LR: learning rate to this structure.
 * L1REG: L1 norm regularization penalty.
 * L2REG: weight decay.


## License
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
