import numpy as np
import torch
from torch.autograd import grad
from Batch import batch_pad
import sys
import time
def train(train_dict, val_dict, model, SSE, SAE, opt_method, optimizer, E_coeff, F_coeff, epoch, val_interval, n_val_stop, convergence, is_force, logfile):

    t0 = time.time()
    model_path = 'best_model'
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']

    n_val = 0

    E_cov = convergence['E_cov']
    F_cov = convergence['F_cov']

    t_ids = np.array(list(train_dict.keys()))
    batch_info = batch_pad(train_dict,t_ids)
    b_fp = batch_info['b_fp']
    
    if is_force:
        b_dfpdX = batch_info['b_dfpdX'].view(b_fp.shape[0],
                                             b_fp.shape[1]*b_fp.shape[2],
                                             b_fp.shape[1]*3)
    b_e_mask = batch_info['b_e_mask']
    b_fp.requires_grad = True
    sb_fp = (b_fp - gmin) / (gmax - gmin)
    N_atoms = batch_info['N_atoms'].view(-1)
    b_e = batch_info['b_e'].view(-1)
    b_f = batch_info['b_f'] 

    
    
    sb_e = (b_e - emin) / (emax - emin)
    sb_f = b_f / (emax - emin) 
    t1 = time.time()
    logfile.write(f'Batching takes {t1-t0}.\n')
    
    v_ids = np.array(list(val_dict.keys()))
    v_batch_info = batch_pad(val_dict,v_ids)
    v_b_fp = v_batch_info['b_fp']
    if is_force:
        v_b_dfpdX = v_batch_info['b_dfpdX'].view(v_b_fp.shape[0],
                                                 v_b_fp.shape[1]*v_b_fp.shape[2],
                                                 v_b_fp.shape[1]*3)
    v_b_e_mask = v_batch_info['b_e_mask']
    v_b_fp.requires_grad = True
    v_sb_fp = (v_b_fp - gmin) / (gmax - gmin)
    v_N_atoms = v_batch_info['N_atoms'].view(-1)
    v_b_e = v_batch_info['b_e'].view(-1)

    v_b_f = v_batch_info['b_f']

   
    v_sb_e = (v_b_e - emin) / (emax - emin)
    v_sb_f = v_b_f / (emax - emin) 


    if opt_method == 'lbfgs':
        for i in range(epoch):
            def closure():
                global E_MAE, F_MAE
                optimizer.zero_grad()
                Atomic_Es = model(sb_fp)
                E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
                if is_force:
                    F_predict = get_forces(E_predict, b_fp, b_dfpdX)
                    metrics =  get_metrics(sb_e, sb_f, N_atoms, t_ids,
                                           E_predict, F_predict, SSE, SAE, scaling, b_e_mask)
                    [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                    loss = E_coeff * E_loss + F_coeff * F_loss
                else:
                    metrics =  get_metrics(sb_e, None, N_atoms, t_ids,
                                           E_predict, None, SSE, SAE, scaling, b_e_mask)
                    [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                    loss = E_coeff * E_loss
                
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)

           
            if i % val_interval == 0:
                n_val += 1
                Atomic_Es = model(sb_fp)
                E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
                if is_force:
                    F_predict = get_forces(E_predict, b_fp, b_dfpdX)
                    metrics =  get_metrics(sb_e, sb_f, N_atoms, t_ids,
                                           E_predict, F_predict, SSE, SAE, scaling, b_e_mask)
                    [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                    loss = E_coeff * E_loss + F_coeff * F_loss
                else:
                    metrics =  get_metrics(sb_e, None, N_atoms, t_ids,
                                           E_predict, None, SSE, SAE, scaling, b_e_mask)
                    [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
                    loss = E_coeff * E_loss
                logfile.write(f'{i}, E_RMSE/atom = {E_RMSE}, F_RMSE = {F_RMSE}, loss={loss}\n')
                logfile.write(f'{i}, E_MAE/atom = {E_MAE}, F_MAE = {F_MAE}\n')



                
                v_Atomic_Es = model(v_sb_fp)
                v_E_predict = torch.sum(v_Atomic_Es * v_b_e_mask, dim = [1,2])
                if is_force:
                    v_F_predict = get_forces(v_E_predict, v_b_fp, v_b_dfpdX)
                    v_metrics =  get_metrics(v_sb_e, v_sb_f, v_N_atoms, v_ids,
                                             v_E_predict, v_F_predict, SSE, SAE, scaling, v_b_e_mask)
                    [v_E_loss, v_F_loss, v_E_MAE, v_F_MAE, v_E_RMSE, v_F_RMSE] = v_metrics
                    v_loss = E_coeff * v_E_loss + F_coeff * v_F_loss
                else:
                    v_metrics =  get_metrics(v_sb_e, None, v_N_atoms, v_ids,
                                             v_E_predict, None, SSE, SAE, scaling, v_b_e_mask)
                    [v_E_loss, v_F_loss, v_E_MAE, v_F_MAE, v_E_RMSE, v_F_RMSE] = v_metrics
                    v_loss = E_coeff * v_E_loss

                try:
                    if v_loss < best_v_loss:
                        best_loss = loss
                        best_E_MAE = E_MAE
                        best_F_MAE = F_MAE
                        best_v_loss = v_loss
                        best_v_E_MAE = v_E_MAE
                        best_v_F_MAE = v_F_MAE
                        torch.save(model,model_path)
                        n_val = 1
                except NameError:
                    best_loss = loss
                    best_E_MAE = E_MAE
                    best_F_MAE = F_MAE
                    best_v_loss = v_loss
                    best_v_E_MAE = v_E_MAE
                    best_v_F_MAE = v_F_MAE
                    torch.save(model,model_path)
                    n_val = 1
                    
                logfile.write(f'val, E_RMSE/atom = {v_E_RMSE}, F_RMSE = {v_F_RMSE}\n')
                logfile.write(f'val, E_MAE/atom = {v_E_MAE}, F_MAE = {v_F_MAE}\n')
                logfile.flush()
                if n_val > n_val_stop:
                    break

    t2 = time.time()
    logfile.write(f'Training takes {t2-t0}\n')
    return [best_loss, best_E_MAE, best_F_MAE, best_v_loss, best_v_E_MAE, best_v_F_MAE]





def get_forces(E_predict, b_fp, b_dfpdX):
    b_dEdfp = grad(E_predict,
                   b_fp,
                   grad_outputs=torch.ones_like(E_predict),
                   create_graph = True,
                   retain_graph = True)[0].view(b_fp.shape[0],1,b_fp.shape[1]*b_fp.shape[2])
    F_predict = - torch.bmm(b_dEdfp,b_dfpdX).view(b_fp.shape[0],b_fp.shape[1],3)
    return F_predict




def get_metrics(sb_e, sb_f, N_atoms, ids, E_predict, F_predict, SSE, SAE, scaling, b_e_mask):

    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']
    

    
    E_loss = SSE(sb_e, E_predict / N_atoms) / len(ids)
    E_MAE = SAE(sb_e, E_predict / N_atoms) / len(ids) * (emax - emin)
    E_RMSE = torch.sqrt(E_loss) * (emax - emin)
    if sb_f is None:
        F_loss = 0
        F_MAE = 0
        F_RMSE = 0
    else:
        F_loss = SSE(sb_f, F_predict) / (3 * torch.sum(N_atoms)) 
        F_MAE = SAE(sb_f, F_predict) / (3 * torch.sum(N_atoms)) * (emax - emin)
        F_RMSE = torch.sqrt(F_loss) * (emax - emin)
    return [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE]


def evaluate(data_dict, SSE, SAE, E_coeff, F_coeff, is_force):

    model = torch.load('best_model')
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']

    ids = np.array(list(data_dict.keys()))
    batch_info = batch_pad(data_dict,ids)
    b_fp = batch_info['b_fp']

    if is_force:
        b_dfpdX = batch_info['b_dfpdX'].view(b_fp.shape[0],
                                             b_fp.shape[1]*b_fp.shape[2],
                                             b_fp.shape[1]*3)
    b_e_mask = batch_info['b_e_mask']
    b_fp.requires_grad = True
    sb_fp = (b_fp - gmin) / (gmax - gmin)
    N_atoms = batch_info['N_atoms'].view(-1)
    b_e = batch_info['b_e'].view(-1)
    b_f = batch_info['b_f'] 

    sb_e = (b_e - emin) / (emax - emin)
    sb_f = b_f / (emax - emin) 


    Atomic_Es = model(sb_fp)
    E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
    if is_force:
        F_predict = get_forces(E_predict, b_fp, b_dfpdX)
        metrics =  get_metrics(sb_e, sb_f, N_atoms, ids,
                               E_predict, F_predict, SSE, SAE, scaling, b_e_mask)
        [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
        loss = E_coeff * E_loss + F_coeff * F_loss
    else:
        metrics =  get_metrics(sb_e, None, N_atoms, ids,
                               E_predict, None, SSE, SAE, scaling, b_e_mask)
        [E_loss, F_loss, E_MAE, F_MAE, E_RMSE, F_RMSE] = metrics
        loss = E_coeff * E_loss
    return [loss, E_MAE, F_MAE]
