import sys

sys.path.append("./SimpleNN")

import os
import shutil
from ase.db import connect
import torch
from ContextManager import cd
from preprocess import train_test_split, train_val_split, get_scaling, CV
from preprocess import snn2sav
from NN import MultiLayerNet
from train import train, evaluate
from fp_calculator import set_sym, calculate_fp
import pickle


is_train = True
is_transfer = False
is_force = True
ensemble_training = True

if is_train and is_transfer:
    raise ValueError('train and transfer could not be true at the same time.')

##################################################################################
#Hyperparameters
##################################################################################
E_coeff = 1
if is_force:
    F_coeff = 1
else:
    F_coeff = 0

val_interval = 10
n_val_stop = 10
epoch = 3000

opt_method = 'lbfgs'


if opt_method == 'lbfgs':
    history_size = 100
    lr = 1
    max_iter = 10
    line_search_fn = 'strong_wolfe'

SSE = torch.nn.MSELoss(reduction='sum')
SAE = torch.nn.L1Loss(reduction='sum')

convergence = {'E_cov':0.0005,'F_cov':0.005}

# min_max will scale fingerprints to (0,1)
fp_scale_method = 'min_max'
e_scale_method = 'min_max'


test_percent = 0.1
# Pecentage from train+val
val_percent = 0.1

# Training model configuration 
if ensemble_training:
    SEED = [1]
    n_ensemble = 5
else:
    SEED = [1,2,3]
n_nodes = [5]
activations = [torch.nn.Tanh()]
#n_nodes = []
#activations = []
lr = 1
hp = {'n_nodes': n_nodes, 'activations': activations, 'lr': lr}

####################################################################################################  
# Configuration for train
####################################################################################################
if is_train:
    # The Name of the training
    Name = f'AuPd_slab_1_train'
    for seed in SEED:
        if not os.path.exists(Name+f'-{seed}'):
            os.makedirs(Name+f'-{seed}')
        
    dbfile = f'./db/AuPd_slab_1_train_dft.db'
    db = connect(dbfile)

    elements = ['Pd','Au']
    nelem = len(elements)
    # This is the energy of the metal in its ground state structure
    #if you don't know the energy of the ground state structure,
    # you can set it to None
    #element_energy = torch.tensor([-1.90060294,-10.84460345/2,-5.51410074,-3.71807396,-8.94730881/2,-10.96382467])
    element_energy = None
    weights = [46,79]
    # Atomic number 
    #weights = [3, 14, 28, 29, 32, 42]
    # Allen electronegativity
    #weights = [0.912,1.916,1.88,1.85,1.994,1.47]
    # Covalent radii
    #weights = [1.28,1.11,1.24,1.32,1.2,1.54]


    Gs = [22,24]
    cutoff = 6.0
    g2_etas = [0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
    #g2_etas = [0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1]
    g2_Rses = [0.0]
    g4_etas=[0.01]
    g4_zetas=[1.0, 4.0]
    g4_lambdas=[-1.0, 1.0]
    sym_params = [Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]
    params_set = set_sym(elements, Gs, cutoff,
                         g2_etas=g2_etas, g2_Rses=g2_Rses,
                         g4_etas=g4_etas, g4_zetas = g4_zetas,
                         g4_lambdas= g4_lambdas, weights=weights)
   
    N_sym = params_set[elements[0]]['num']
    
####################################################################################################  
# Configuration for transfer
####################################################################################################
if is_transfer:
    source_Name = 'combined_noNi_e'
    # The Name of the training
    Name = f'combined_Ni_e'
    for seed in SEED:
        if not os.path.exists(Name+f'-{seed}'):
            os.makedirs(Name+f'-{seed}')
        
    dbfile = f'./db/Ni.db'
    db = connect(dbfile)

    elements = ['Li', 'Si', 'Ni', 'Cu', 'Ge', 'Mo']
    nelem = len(elements)
    # This is the energy of the metal in its ground state structure
    #if you don't know the energy of the ground state structure,
    # you can set it to None
    element_energy = torch.tensor([-1.90060294,-10.84460345/2,-5.51410074,-3.71807396,-8.94730881/2,-10.96382467])


    # Atomic number 
    #weights = [3, 14, 28, 29, 32, 42]
    # Allen electronegativity
    weights = [0.912,1.916,1.88,1.85,1.994,1.47]
    # Covalent radii
    #weights = [1.28,1.11,1.24,1.32,1.2,1.54]


    
    

####################################################################################################  
# Train
####################################################################################################
# if is_train:
#     for seed in SEED:
#         # This use the context manager to operate in the data directory
#         with cd(Name+f'-{seed}'):
#             pickle.dump(sym_params, open("sym_params.sav", "wb"))
#             logfile = open('log.txt','w+')
#             resultfile = open('result.txt','w+')
            
#             if os.path.exists('test.sav'):
#                 logfile.write('Did not calculate symfunctions.\n')
#             else:
#                 data_dict = snn2sav(db, Name, elements, params_set,
#                                     element_energy=element_energy)
#                 train_dict = train_test_split(data_dict,1-test_percent,seed=seed)
#                 train_val_split(train_dict,1-val_percent,seed=seed)
                
#             logfile.flush()
            
#             train_dict = torch.load('final_train.sav')
#             val_dict = torch.load('final_val.sav')
#             test_dict = torch.load('test.sav')
            
#             scaling = get_scaling(train_dict, fp_scale_method, e_scale_method)
            
            
#             n_nodes = hp['n_nodes']
#             activations = hp['activations']
#             lr = hp['lr']
#             model = MultiLayerNet(N_sym, n_nodes, activations, nelem, scaling=scaling)
#             if opt_method == 'lbfgs':
#                 optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,
#                                               max_iter=max_iter, history_size=history_size,
#                                               line_search_fn=line_search_fn)
             
#             results = train(train_dict, val_dict,
#                             model,
#                             opt_method, optimizer,
#                             E_coeff, F_coeff,
#                             epoch, val_interval,
#                             n_val_stop,
#                             convergence, is_force,
#                             logfile)
#             [loss, E_MAE, F_MAE, v_loss, v_E_MAE, v_F_MAE] = results
            
#             test_results = evaluate(test_dict, E_coeff, F_coeff, is_force)
#             [test_loss, test_E_MAE, test_F_MAE] =test_results
#             resultfile.write(f'Hyperparameter: n_nodes = {n_nodes}, activations = {activations}, lr = {lr}\n')
#             resultfile.write(f'loss = {loss}, E_MAE = {E_MAE}, F_MAE = {F_MAE}.\n')
#             resultfile.write(f'v_loss = {v_loss}, v_E_MAE = {v_E_MAE}, v_F_MAE = {v_F_MAE}.\n')
#             resultfile.write(f'test_loss = {test_loss}, test_E_MAE = {test_E_MAE}, test_F_MAE = {test_F_MAE}.\n')
            

#             logfile.close()
#             resultfile.close()
if is_train:
    for seed in SEED:
        # This use the context manager to operate in the data directory
        with cd(Name+f'-{seed}'):
            pickle.dump(sym_params, open("sym_params.sav", "wb"))
            logfile = open('log.txt','w+')
            resultfile = open('result.txt','w+')
            
            if os.path.exists('test.sav'):
                logfile.write('Did not calculate symfunctions.\n')
            else:
                data_dict = snn2sav(db, Name, elements, params_set,
                                    element_energy=element_energy)
                train_dict = train_test_split(data_dict,1-test_percent,seed=seed)
                train_val_split(train_dict,1-val_percent,seed=seed)
                
            logfile.flush()
            
            train_dict = torch.load('final_train.sav')
            val_dict = torch.load('final_val.sav')
            test_dict = torch.load('test.sav')
            
            scaling = get_scaling(train_dict, fp_scale_method, e_scale_method)
            
            
            n_nodes = hp['n_nodes']
            activations = hp['activations']
            lr = hp['lr']
            if ensemble_training:
                for ensemble in range(n_ensemble):
                    model = MultiLayerNet(N_sym, n_nodes, activations, nelem, scaling=scaling)
                    if opt_method == 'lbfgs':
                        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,
                                                      max_iter=max_iter, history_size=history_size,
                                                      line_search_fn=line_search_fn)
                    logfile.write(f'Ensemble {ensemble}\n')
                    model_path = f'best_model-{ensemble}'
                    results = train(train_dict, val_dict,
                                    model, model_path,
                                    opt_method, optimizer,
                                    E_coeff, F_coeff,
                                    epoch, val_interval,
                                    n_val_stop,
                                    convergence, is_force,
                                    logfile)
                    [loss, E_MAE, F_MAE, v_loss, v_E_MAE, v_F_MAE] = results
                    
                    test_results = evaluate(test_dict, model_path, E_coeff, F_coeff, is_force)
                    [test_loss, test_E_MAE, test_F_MAE] =test_results
                    resultfile.write(f'Ensemble {ensemble}\n')
                    resultfile.write(f'Hyperparameter: n_nodes = {n_nodes}, activations = {activations}, lr = {lr}\n')
                    resultfile.write(f'loss = {loss}, E_MAE = {E_MAE}, F_MAE = {F_MAE}.\n')
                    resultfile.write(f'v_loss = {v_loss}, v_E_MAE = {v_E_MAE}, v_F_MAE = {v_F_MAE}.\n')
                    resultfile.write(f'test_loss = {test_loss}, test_E_MAE = {test_E_MAE}, test_F_MAE = {test_F_MAE}.\n')
            

            else:
                 model = MultiLayerNet(N_sym, n_nodes, activations, nelem, scaling=scaling)
                 if opt_method == 'lbfgs':
                     optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,
                                                   max_iter=max_iter, history_size=history_size,
                                                   line_search_fn=line_search_fn)
                 model_path = 'best_model'
                 results = train(train_dict, val_dict,
                                 model, model_file,
                                 opt_method, optimizer,
                                 E_coeff, F_coeff,
                                 epoch, val_interval,
                                 n_val_stop,
                                 convergence, is_force,
                                 logfile)
                 [loss, E_MAE, F_MAE, v_loss, v_E_MAE, v_F_MAE] = results
                 
                 test_results = evaluate(test_dict, model_path, E_coeff, F_coeff, is_force)
                 [test_loss, test_E_MAE, test_F_MAE] =test_results
                 resultfile.write(f'Hyperparameter: n_nodes = {n_nodes}, activations = {activations}, lr = {lr}\n')
                 resultfile.write(f'loss = {loss}, E_MAE = {E_MAE}, F_MAE = {F_MAE}.\n')
                 resultfile.write(f'v_loss = {v_loss}, v_E_MAE = {v_E_MAE}, v_F_MAE = {v_F_MAE}.\n')
                 resultfile.write(f'test_loss = {test_loss}, test_E_MAE = {test_E_MAE}, test_F_MAE = {test_F_MAE}.\n')
                 
                 
            logfile.close()
            resultfile.close()


####################################################################################################  
# Transfer
####################################################################################################
if is_transfer:
    for seed in SEED:
        # This use the context manager to operate in the data directory
        with cd(source_Name+f'-{seed}'):
            model = torch.load('best_model')
            sym_params = pickle.load(open( "sym_params.sav", "rb" ))
            [Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, _, _, _]=sym_params
            sym_params = [Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]
            params_set = set_sym(elements, Gs, cutoff,
                         g2_etas=g2_etas, g2_Rses=g2_Rses,
                         g4_etas=g4_etas, g4_zetas = g4_zetas,
                         g4_lambdas= g4_lambdas, weights=weights)
   
            N_sym = params_set[elements[0]]['num']
        with cd(Name+f'-{seed}'):
            pickle.dump(sym_params, open("sym_params.sav", "wb"))
            logfile = open('log.txt','w+')
            resultfile = open('result.txt','w+')
            
            
            if os.path.exists('test.sav'):
                logfile.write('Did not calculate symfunctions.\n')
            else:
                data_dict = snn2sav(db, Name, elements, params_set,
                                    element_energy=element_energy)
                train_dict = train_test_split(data_dict,1-test_percent,seed=seed)
                train_val_split(train_dict,1-val_percent,seed=seed)
                
            logfile.flush()
            
            train_dict = torch.load('final_train.sav')
            val_dict = torch.load('final_val.sav')
            test_dict = torch.load('test.sav')
            
            
            
            #n_nodes = hp['n_nodes']
            #activations = hp['activations']
            lr = hp['lr']
            for param in model.parameters():
                param.requires_grad = False
            H = model.net[-1].in_features
            model.net[-1] = torch.nn.Linear(H, nelem)
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())

            if opt_method == 'lbfgs':
                optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,
                                              max_iter=max_iter, history_size=history_size,
                                              line_search_fn=line_search_fn)
             
            results = train(train_dict, val_dict,
                            model,
                            opt_method, optimizer,
                            E_coeff, F_coeff,
                            epoch, val_interval,
                            n_val_stop,
                            convergence, is_force,
                            logfile)
            [loss, E_MAE, F_MAE, v_loss, v_E_MAE, v_F_MAE] = results
            
            test_results = evaluate(test_dict, E_coeff, F_coeff, is_force)
            [test_loss, test_E_MAE, test_F_MAE] =test_results
            resultfile.write(f'Hyperparameter: n_nodes = {n_nodes}, activations = {activations}, lr = {lr}\n')
            resultfile.write(f'loss = {loss}, E_MAE = {E_MAE}, F_MAE = {F_MAE}.\n')
            resultfile.write(f'v_loss = {v_loss}, v_E_MAE = {v_E_MAE}, v_F_MAE = {v_F_MAE}.\n')
            resultfile.write(f'test_loss = {test_loss}, test_E_MAE = {test_E_MAE}, test_F_MAE = {test_F_MAE}.\n')
            

            logfile.close()
            resultfile.close()


