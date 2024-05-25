import torch
import torch.nn as nn
import math
from typing import Optional


class MultiOmicsLayer(nn.Module):
    """
    MultiOmicsLayer represents a neural network layer for modeling complex
    relationships in multi-omics data, including:
    - Genotypic variations (e.g., SNPs) affecting gene expression
    - Gene-protein regulatory networks
    - Protein-protein interaction networks
    - Integration of gene expression and protein levels to model proteoforms 
      or protein complexes

    Args:
        in_dims (int):  Number of dimensions of input features.
        out_dims (int): Number of dimensions of output features.
    bias (bool): If True, adds a learnable bias to the output. Default: True.
    """
    def __init__(self, in_dims: int, out_dims: int, bias: bool = True) -> None:
        super(MultiOmicsLayer, self).__init__()
        self.in_dims  = in_dims
        self.out_dims = out_dims
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dims))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """
        Initialize parameters using Kaiming uniform distribution.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            input (torch.Tensor): Input tensor.
            adj_matrix (torch.Tensor): Adjacency matrix tensor.

        Returns:
            torch.Tensor: Output tensor after applying the transformation.
        """
        return input.matmul(self.weight.t() * adj_matrix) + self.bias



class TransferModuleBase(nn.Module):
    """
    Base class for transfer modules with a common architecture.
    
    Args:
        adj_matrix (torch.Tensor): Adjacency matrix representing relationships
                                   between input and output features.
        in_dims (int): Number of dimensions of input features.
        out_dims (int): Number of dimensions of output features.
        H1 (int): Size of the first hidden layer.
        H2 (Optional[int]): Size of the second hidden layer (optional).
        D_out (int): Size of the output layer.
        drop_rate (float): Dropout rate for regularization.
    """

    def __init__(self, adj_matrix: torch.Tensor, 
                 in_dims: int, out_dims: int, 
                 H1: int, H2: Optional[int], 
                 D_out: int, drop_rate: float) -> None:
        super(TransferModuleBase, self).__init__()
        self.adj_matrix = adj_matrix
        self.SingleOmic = MultiOmicsLayer(in_dims, out_dims)
        self.dropout1 = nn.Dropout(drop_rate)     # Dropout after SingleOmic
        self.hidden1 = nn.Linear(out_dims, H1)    # First hidden layer
        self.dropout2 = nn.Dropout(drop_rate)     # Dropout after 1st hidden
        self.H2 = H2
        if H2 is not None:
            self.hidden2 = nn.Linear(H1, H2)      # Second hidden layer
            self.dropout3 = nn.Dropout(drop_rate) # Dropout after 2nd hidden
            self.output = nn.Linear(H2, D_out)    # Output layer
        else:
            self.output = nn.Linear(H1, D_out)    # Output layer

    def forward(self, in_mat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            in_mat (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the module.
        """
        t1 = self.SingleOmic(in_mat, self.adj_matrix).relu()
        t1 = self.dropout1(t1)     # Apply dropout after SingleOmic
        h1 = self.hidden1(t1).relu()
        h1 = self.dropout2(h1)     # Apply dropout after 1st hidden layer
        if self.H2 is not None:
            h2 = self.hidden2(h1).relu()
            h2 = self.dropout3(h2) # Apply dropout after 2bd hidden layer
            y_pred = self.output(h2).sigmoid()
        else:
            y_pred = self.output(h1).sigmoid()
        return y_pred

# Inherit ModuleBase for pretrain moduels
class TransferModuleSNP2Layer(TransferModuleBase):
    """
    Transfer module for SNP (Single Nucleotide Polymorphism) to gene layer.

    Inherits from TransferModuleBase and passes specific parameters.
    """
    def __init__(self, adj_snp_gen: torch.Tensor, 
                 SNP_in: int, Gen_out: int, 
                 H1: int, H2: Optional[int], 
                 D_out: int, drop_rate: float) -> None:
        super(TransferModuleSNP2Layer, self).__init__(
            adj_snp_gen, SNP_in, Gen_out, H1, H2, D_out, drop_rate)

class TransferModuleGene2Layer(TransferModuleBase):
    """
    Transfer module for gene to protein layer mapping.

    Inherits from TransferModuleBase and passes specific parameters.
    """
    def __init__(self, adj_gen_pro: torch.Tensor, 
                 Gen_in: int, Pro_out: int, 
                 H1: int, H2: Optional[int], 
                 D_out: int, drop_rate: float) -> None:
        super(TransferModuleGene2Layer, self).__init__(
            adj_gen_pro, Gen_in, Pro_out, H1, H2, D_out, drop_rate)
        

class TransferModuleProtein2Layer(TransferModuleBase):
    """
    Transfer module for protein to protein layer mapping.

    Inherits from TransferModuleBase and passes specific parameters.
    """
    def __init__(self, adj_pro_pro: torch.Tensor, 
                 Pro_in: int, Pro_out: int, 
                 H1: int, H2: Optional[int], 
                 D_out: int, drop_rate: float) -> None:
        super(TransferModuleProtein2Layer, self).__init__(
            adj_pro_pro, Pro_in, Pro_out, H1, H2, D_out, drop_rate)
                

class TransFuse(nn.Module):
    """
    Base class for transfer modules with a common architecture.

    Args:
        adj_snp_gen (torch.Tensor): Adjacency matrix for SNP to gene 
                                    connections.
        adj_gen_pro (torch.Tensor): Adjacency matrix for gene to protein 
                                    connections.
        adj_genpro_pro (torch.Tensor): Adjacency matrix for combin-gene-protein
                                       to protein connections.
        adj_pro_pro (torch.Tensor): Adjacency matrix for protein to protein 
                                    connections.
        n_snp (int): Number of dimensions of input SNP.
        n_gen (int): Number of dimensions of input gene.
        n_pro (int): Number of dimensions of input protein.
        H1 (int): Size of the first hidden layer.
        H2 (Optional[int]): Size of the second hidden layer (optional).
        D_out (int): Size of the output layer.
        drop_rate (float): Dropout rate for regularization.
    """
    def __init__(self, adj_snp_gen: torch.Tensor, adj_gen_pro: torch.Tensor, 
                 adj_genpro_pro: torch.Tensor, adj_pro_pro: torch.Tensor,
                 n_snp: int, n_gen: int, n_pro: int, 
                 H1: int, H2: Optional[int],
                 D_out: int, drop_rate: float) -> None:
        super(TransFuse, self).__init__()
        self.adj_snp2gen = adj_snp_gen
        self.adj_gen2pro = adj_gen_pro
        self.adj_genpro2pro = adj_genpro_pro
        self.adj_pro2pro = adj_pro_pro
        
        self.n_snp = n_snp
        self.n_gen = n_gen
        self.n_pro = n_pro
        
        self.LayerSnp2gen = MultiOmicsLayer(n_snp, n_gen)
        self.LayerGen2pro = MultiOmicsLayer(n_gen, n_pro)
        self.LayerBridge  = MultiOmicsLayer(n_gen+n_pro, n_pro)
        self.LayerPro2pro = MultiOmicsLayer(n_pro, n_pro)
        
        self.dropout1 = nn.Dropout(drop_rate)     # Dropout after fusion layer
        self.hidden1 = nn.Linear(n_pro+n_pro, H1) # First hidden layer
        self.dropout2 = nn.Dropout(drop_rate)     # Dropout after 1st hidden
        self.H2 = H2
        if H2 is not None:
            self.hidden2 = nn.Linear(H1, H2)      # Second hidden layer
            self.dropout3 = nn.Dropout(drop_rate) # Dropout after 2nd hidden
            self.output = nn.Linear(H2, D_out)    # Output layer
        else:
            self.output = nn.Linear(H1, D_out)    # Output layer

    def forward(self, in_mat):
        """
        Forward pass of the module.

        Args:
            in_mat (Tensor): Input feature matrix in tensor format.

        Returns:
            Tensor: Output tensor after passing through the module.
        """
        n_features = self.n_pro+self.n_gen+self.n_snp
        
        # pretrain module SNP to Gene
        out_snp = self.LayerSnp2gen(
            in_mat[:, (self.n_pro+self.n_gen):n_features],
            self.adj_snp2gen).relu()
        
        # pretrain module Gene to Protein
        out_gene = self.LayerGen2pro(
            in_mat[:, self.n_pro:(self.n_pro+self.n_gen)],
            self.adj_gen2pro).relu()
        
        # bridge module, to be trained
        in_bridge = torch.cat((out_snp, out_gene),1)
        out_bridge = self.LayerBridge(in_bridge, self.adj_genpro2pro).relu()
        
        # pretrain module Protein to Protein
        out_protein = self.LayerPro2pro(
            in_mat[:, 0:self.n_pro],
            self.adj_pro2pro).relu()
        
        # concatenate outputs of bridge module and pretrain protein module
        out_fuse = torch.cat((out_bridge, out_protein),1)
        out_fuse = self.dropout1(out_fuse) # Apply dropout after fusion layer
        h1 = self.hidden1(out_fuse).relu()
        h1 = self.dropout2(h1)     # Apply dropout after 1st hidden layer
        if self.H2 is not None:
            h2 = self.hidden2(h1).relu()
            h2 = self.dropout3(h2) # Apply dropout after 2bd hidden layer
            y_pred = self.output(h2).sigmoid()
        else:
            y_pred = self.output(h1).sigmoid()
        return y_pred
    
