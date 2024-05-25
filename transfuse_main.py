"""
Usage:
    Example running in command line with required command arguments:
    	python transfuse_main.py /the_path_to_your_dataset/ baseline
"""

import os
import pandas as pd
import numpy as np
import argparse
from utils import set_random_seeds
from train import (pretraining, train_baseline_model, test_baseline_model)
from model import (TransferModuleSNP2Layer, TransferModuleGene2Layer,
                   TransferModuleProtein2Layer)
import torch


def main():
    parser = argparse.ArgumentParser(
        description = 'A script with command-line arguments'
        )
    
    # Add a positinoal argument 'folder' for the main input
    parser.add_argument(
        'folder',
        help = 'Specify the folder where all .csv files located.'
        )
    
    # Add a positinoal argument 'modle' for the main input
    parser.add_argument(
        'model',
        choices = ['pretrain', 'baseline', 'transferweight', 'fine-tune'],
        help = 'Specify training on baseline or apply transfer learning.'
        )
    
    args =  parser.parse_args()
    
    # Access the argument
    folder_value = args.folder
    model_value = args.model
    
    current_path = os.path.abspath(os.path.dirname(__file__))
    print(f'Current path is {current_path}.')
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adj_snp2gen_tmp = pd.read_csv(folder_value + 'adj_snp_gen.csv')
    adj_snp2gen_df  = adj_snp2gen_tmp.set_index('Row')
    adj_snp2gen_np  = np.array(adj_snp2gen_df)
    
    adj_gen2pro_tmp = pd.read_csv(folder_value + 'adj_gen_pro.csv')
    adj_gen2pro_df = adj_gen2pro_tmp.set_index('Row')
    adj_gen2pro_np  = np.array(adj_gen2pro_df)
    
    adj_bridge_tmp = pd.read_csv(folder_value + 'adj_genpro_pro.csv')
    adj_bridge_df  = adj_bridge_tmp.set_index('Row')
    adj_bridge_np  = np.array(adj_bridge_df)
    
    adj_pro2pro_tmp = pd.read_csv(folder_value + 'adj_pro_pro.csv')
    adj_pro2pro_df = adj_pro2pro_tmp.set_index('Row')
    adj_pro2pro_np = np.array(adj_pro2pro_df)
    
    # Define the shape of the network
    adj_snp2gen = torch.from_numpy(adj_snp2gen_np).float().to(device)
    adj_gen2pro = torch.from_numpy(adj_gen2pro_np).float().to(device)
    adj_bridge  = torch.from_numpy(adj_bridge_np).float().to(device)
    adj_pro2pro = torch.from_numpy(adj_pro2pro_np).float().to(device)
    
    # multi-omic features dimensions
    n_snp, n_gen, n_pro = 822, 743, 186
    
    if model_value == 'pretrain':
        
        # load the input data
        train_x_snp = pd.read_csv(folder_value + 'pretrain_X_snp_train.csv')
        train_y_snp = pd.read_csv(folder_value + 'pretrain_y_snp_train.csv')
        valid_x_snp = pd.read_csv(folder_value + 'pretrain_X_snp_test.csv')
        valid_y_snp = pd.read_csv(folder_value + 'pretrain_y_snp_test.csv')
        
        train_x_gene = pd.read_csv(folder_value + 'pretrain_X_gene_train.csv')
        train_y_gene = pd.read_csv(folder_value + 'pretrain_y_gene_train.csv')
        valid_x_gene = pd.read_csv(folder_value + 'pretrain_X_gene_test.csv')
        valid_y_gene = pd.read_csv(folder_value + 'pretrain_y_gene_test.csv')
        
        train_x_protein = pd.read_csv(folder_value + 
                                      'pretrain_X_protein_train.csv')
        train_y_protein = pd.read_csv(folder_value + 
                                      'pretrain_y_protein_train.csv')
        valid_x_protein = pd.read_csv(folder_value + 
                                      'pretrain_X_protein_test.csv')
        valid_y_protein = pd.read_csv(folder_value + 
                                      'pretrain_y_protein_test.csv')

        # pre-set hyper-parameters
        LR, L1REG, L2REG = 0.001, 0.001, 0.002
        epochs, BS, n_seed, dropout_rate = 30, 32, 66, 0.5
        ythresh = 0.5
        betas_range = (0.9, 0.999)
        # pre-set Network structure
        H1, H2, D_out = 16, None, 1 
        
        set_random_seeds(n_seed)
        model_snp, pretrain_weights_snp, loss, accuracy = pretraining(
                        train_x_snp, train_y_snp, 
                        valid_x_snp, valid_y_snp,
                        TransferModuleSNP2Layer, adj_snp2gen,
                        n_snp, n_gen,
                        H1, H2, D_out, dropout_rate,
                        LR, L1REG, L2REG, 
                        betas_range, ythresh, 
                        BS, epochs, device)
        torch.save(pretrain_weights_snp,
                   f'{folder_value}/pretrain_snp_weights.pth')
        
        set_random_seeds(n_seed)
        model_gene, pretrain_weights_gene, loss, accuracy = pretraining(
                        train_x_gene, train_y_gene, 
                        valid_x_gene, valid_y_gene,
                        TransferModuleGene2Layer, adj_gen2pro,
                        n_gen, n_pro,
                        H1, H2, D_out, dropout_rate,
                        LR, L1REG, L2REG, 
                        betas_range, ythresh, 
                        BS, epochs, device)
        torch.save(pretrain_weights_gene,
                   f'{folder_value}/pretrain_gene_weights.pth')
        
        set_random_seeds(n_seed)
        model_protein, pretrain_weights_protein, loss, accuracy = pretraining(
                        train_x_protein, train_y_protein, 
                        valid_x_protein, valid_y_protein,
                        TransferModuleProtein2Layer, adj_pro2pro,
                        n_pro, n_pro,
                        H1, H2, D_out, dropout_rate,
                        LR, L1REG, L2REG, 
                        betas_range, ythresh, 
                        BS, epochs, device)
        torch.save(pretrain_weights_protein,
                   f'{folder_value}/pretrain_protein_weights.pth')
        
        # Exit the main function after pretraining
        return
    
    elif model_value == 'baseline':
        transfer_flag = False
        finetune_flag = False
    
    elif model_value == 'transferweight':
        transfer_flag = True
        finetune_flag = False
    
    elif model_value == 'fine-tune':
        transfer_flag = True
        finetune_flag = True
    
    
    pretrain_weights_snp = torch.load(
                    f'{folder_value}/pretrain_snp_weights.pth')
    pretrain_weights_gene = torch.load(
                    f'{folder_value}/pretrain_gene_weights.pth')
    pretrain_weights_protein = torch.load(
                    f'{folder_value}/pretrain_protein_weights.pth')
    
    # pre-set hyper-parameters
    LR, L1REG, L2REG = 0.00006, 0.005, 0.0005
    epochs, BS, n_seed, dropout_rate = 50, 32, 66, 0.5
    ythresh = 0.5
    betas_range = (0.9, 0.999)
    # pre-set Network structure
    H1, H2, D_out = 32, None, 1 
    
    adj_matrices = (adj_snp2gen, adj_gen2pro, adj_bridge, adj_pro2pro)
    dimensions = (n_snp, n_gen, n_pro, H1, H2, D_out)
    pretrained_weights = (pretrain_weights_snp,
                          pretrain_weights_gene,
                          pretrain_weights_protein)
                    
    # define empty list to store each fold's result
    fold_results = []
    
    for fold in range(1, 6):
    
        set_random_seeds(n_seed)
        print(f"Training on Fold {fold}")
        
        if model_value == 'fine-tune':
            LR = 0.000006
            train_model_transfer = torch.load(
                f'{folder_value}/model_train_{fold}.pth')
            pretrained_weights = (
                train_model_transfer.LayerSnp2gen.state_dict(),
                train_model_transfer.LayerGen2pro.state_dict(),
                train_model_transfer.LayerPro2pro.state_dict())
          
        # Call the train_baseline_model function
        val_metrics, trained_model = \
            train_baseline_model(
              fold, adj_matrices, dimensions, dropout_rate, LR, L1REG, 
              L2REG, betas_range, ythresh, BS, epochs, device, n_seed,
              pretrained_weights, transfer_flag, finetune_flag, folder_value
            )
            
        # Store the results and models for each fold
        fold_results.append(val_metrics)
        
        # save the trained model
        if model_value == 'fine-tune':
            torch.save(trained_model, 
                       f'{folder_value}/model_ft_train_{fold}.pth')
        else:
            torch.save(trained_model, 
                       f'{folder_value}/model_train_{fold}.pth')
    
    # Process and display the results after cross-validation
    v_avg_accuracy = np.mean([result['accuracy'] 
                        for result in fold_results])
    v_avg_f1 = np.mean([result['f1_score'] 
                        for result in fold_results])
    v_avg_recall = np.mean([result['recall'] 
                        for result in fold_results])
    v_avg_precision = np.mean([result['precision'] 
                        for result in fold_results])
    v_avg_specificity = np.mean([result['specificity'] 
                        for result in fold_results])
    v_avg_auc_score = np.mean([result['auc_score'] 
                        for result in fold_results])
    
    print(f"Validation Average Accuracy   : {v_avg_accuracy:.4f}")
    print(f"Validation Average F1 Score   : {v_avg_f1:.4f}")
    print(f"Validation Average Recall     : {v_avg_recall:.4f}")
    print(f"Validation Average Precision  : {v_avg_precision:.4f}")
    print(f"Validation Average Specificity: {v_avg_specificity:.4f}")
    print(f"Validation Average AUC Score  : {v_avg_auc_score:.4f}")

    print('--------------------------------------')
    print('-----------Testing result-------------')
    print('--------------------------------------')
    
    # define empty list to store each fold's result
    fold_results = []
    
    # Loop over each fold for testing
    for fold in range(1, 6):
    
        set_random_seeds(n_seed)
        
        if model_value == 'fine-tune':
            test_model = torch.load( 
                       f'{folder_value}/model_ft_train_{fold}.pth')
        else:
            test_model = torch.load( 
                       f'{folder_value}/model_train_{fold}.pth')
    
        # Call the train_baseline_model function
        test_metrics = test_baseline_model(
              fold, test_model, ythresh, device, folder_value
            )
    
        print(f"Fold {fold} test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Fold {fold} test f1: {test_metrics['f1_score']:.4f}")
    
        # Store the results and models for each fold
        fold_results.append(test_metrics)
    
    # Process and display the results after cross-validation
    t_avg_accuracy = np.mean([result['accuracy'] 
                        for result in fold_results])
    t_avg_f1 = np.mean([result['f1_score'] 
                        for result in fold_results])
    t_avg_recall = np.mean([result['recall'] 
                        for result in fold_results])
    t_avg_precision = np.mean([result['precision'] 
                        for result in fold_results])
    t_avg_specificity = np.mean([result['specificity'] 
                        for result in fold_results])
    t_avg_auc_score = np.mean([result['auc_score'] 
                        for result in fold_results])
    
    print(f"Testing Average Accuracy   : {t_avg_accuracy:.4f}")
    print(f"Testing Average F1 Score   : {t_avg_f1:.4f}")
    print(f"Testing Average Recall     : {t_avg_recall:.4f}")
    print(f"Testing Average Precision  : {t_avg_precision:.4f}")
    print(f"Testing Average Specificity: {t_avg_specificity:.4f}")
    print(f"Testing Average AUC Score  : {t_avg_auc_score:.4f}")


if __name__ == '__main__':
    try:
        main()
    except argparse.ArgumentError as e:
        print('To run the code, use the following command in your terminal:')
        print('python main.py data_path baseline')
        print(f'Error: {e}')