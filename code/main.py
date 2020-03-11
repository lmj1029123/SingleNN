
import torch
from ContextManager import cd
from preprocess import get_scaling
from NN import TwoLayerNet
from train import train, evaluate


####################################################################################################
# Mode of operation
####################################################################################################
is_train = True
is_transfer = False

is_force = True


if is_train and is_transfer:
    raise ValueError('train and transfer could not be true at the same time.')


####################################################################################################
# Optimization detail
####################################################################################################

# This are the coefficients for energy and forces loss on the objective function
E_coeff = 100
if is_force:
    F_coeff = 1
else:
    F_coeff = 0

# This indicates how often does the optimizer checks the validation error 
val_interval = 1

# This determines the number validation error check that will be done before the optimizer stops, if the validation error does not decrease.

n_val_stop = 10

opt_method = 'lbfgs'

# Not important here, they are placeholder for Adam optimizer but not used in the supporting information.
mini_batch = False
batch_size = 32


if opt_method == 'lbfgs':
    history_size = 100
    lr = 1
    max_iter = 10
    line_search_fn = 'strong_wolfe'
    if is_train == True:
        epoch = 3000
    elif is_transfer == True:
        epoch = 1000
 


SSE = torch.nn.MSELoss(reduction='sum')
SAE = torch.nn.L1Loss(reduction='sum')

convergence = {'E_cov':0.0005,'F_cov':0.005}

# min_max will scale fingerprints to (0,1)
fp_scale_method = 'min_max'
e_scale_method = 'min_max'



seed = 1
H1 = 20
H2 = 20
# This is specified here because we used 14 symfunction and 6 elements in the paper
N_sym = 14
nelem = 6
lr = 1


Name = 'combined'

if is_transfer:
    source_Name = 'Cu-1'
    with cd(source_Name):
        model = torch.load('best_model')





with cd(Name+f'-{seed}'):
    
    logfile = open('log.txt','w+')
    resultfile = open('result.txt','w+')
    train_dict = torch.load('final_train.sav')
    val_dict = torch.load('final_val.sav')
    test_dict = torch.load('test.sav')
    scaling = get_scaling(train_dict, fp_scale_method, e_scale_method)

    if is_train:
        model = TwoLayerNet(N_sym, H1, H2, nelem, scaling=scaling)

    if is_transfer:
        for param in model.parameters():
            param.requires_grad = False
        H2 = model.linear3.in_features
        model.linear3 = torch.nn.Linear(H2, nelem)
        # model.scaling = scaling
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        
    if opt_method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,
                                      max_iter=max_iter, history_size=history_size,
                                    line_search_fn=line_search_fn)
        
    [loss, E_MAE, F_MAE, v_loss, v_E_MAE, v_F_MAE]= train(train_dict, val_dict,
                                                          model, SSE, SAE,
                                                          opt_method, optimizer,
                                                          mini_batch, batch_size,
                                                          E_coeff, F_coeff,
                                                          epoch, val_interval,
                                                          n_val_stop,
                                                          convergence, is_force,
                                                          logfile)
            
    [test_loss, test_E_MAE, test_F_MAE] = evaluate(test_dict, SSE, SAE, E_coeff, F_coeff, is_force)
    
    resultfile.write(f'loss = {loss}, E_MAE = {E_MAE}, F_MAE = {F_MAE}.\n')
    resultfile.write(f'v_loss = {v_loss}, v_E_MAE = {v_E_MAE}, v_F_MAE = {v_F_MAE}.\n')
    resultfile.write(f'test_loss = {test_loss}, test_E_MAE = {test_E_MAE}, test_F_MAE = {test_F_MAE}.\n')
        
    
    logfile.close()
    resultfile.close()




