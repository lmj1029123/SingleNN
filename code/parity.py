import matplotlib
matplotlib.use('Agg') 



import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import torch
from ContextManager import cd

from Batch import batch_pad
import torch
import numpy as np
from torch.autograd import grad
import pandas as pd

Name = 'combined-1'
is_force = True

with cd(Name):
    train_dict = torch.load('final_train.sav')
    val_dict = torch.load('test.sav')
    model = torch.load('best_model')
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']

    t_ids = np.array(list(train_dict.keys()))
    batch_info = batch_pad(train_dict,t_ids)
    b_fp = batch_info['b_fp']
    b_e_mask = batch_info['b_e_mask']
    if is_force:
        b_fp.requires_grad = True
    sb_fp = (b_fp - gmin) / (gmax - gmin)
    N_atoms = batch_info['N_atoms'].view(-1)
    b_e = batch_info['b_e'].view(-1)
    if is_force:
        b_f = batch_info['b_f'] 

    v_ids = np.array(list(val_dict.keys()))
    v_batch_info = batch_pad(val_dict,v_ids)
    v_b_fp = v_batch_info['b_fp']
    v_b_e_mask = v_batch_info['b_e_mask']
    if is_force:
        v_b_fp.requires_grad = True
    v_sb_fp = (v_b_fp - gmin) / (gmax - gmin)
    v_N_atoms = v_batch_info['N_atoms'].view(-1)
    v_b_e = v_batch_info['b_e'].view(-1)
    if is_force:
        v_b_f = v_batch_info['b_f'] 

    Atomic_Es = model(sb_fp)
    E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
    E_predict = E_predict * (emax-emin) + emin * N_atoms
    if is_force:
        b_dEdfp = grad(E_predict,
                       b_fp,
                       grad_outputs=torch.ones_like(E_predict),
                       create_graph = True,
                       retain_graph = True)[0].view(b_fp.shape[0],1,b_fp.shape[1]*b_fp.shape[2])
        b_dfpdX = batch_info['b_dfpdX'].view(b_fp.shape[0],b_fp.shape[1]*b_fp.shape[2],b_fp.shape[1]*3)
        F_predict = - torch.bmm(b_dEdfp,b_dfpdX).view(b_fp.shape[0],b_fp.shape[1],3)

    
    v_Atomic_Es = model(v_sb_fp)
    v_E_predict = torch.sum(v_Atomic_Es * v_b_e_mask, dim = [1,2])
    v_E_predict = v_E_predict * (emax-emin) + emin * v_N_atoms
    if is_force:
        v_b_dEdfp = grad(v_E_predict,
                         v_b_fp,
                         grad_outputs=torch.ones_like(v_E_predict),
                         create_graph = True,
                         retain_graph = True)[0].view(v_b_fp.shape[0],1,
                                                      v_b_fp.shape[1]*v_b_fp.shape[2])
        v_b_dfpdX = v_batch_info['b_dfpdX'].view(v_b_fp.shape[0],
                                                 v_b_fp.shape[1]*v_b_fp.shape[2],
                                                 v_b_fp.shape[1]*3)
    
        v_F_predict = - torch.bmm(v_b_dEdfp,v_b_dfpdX).view(v_b_fp.shape[0],v_b_fp.shape[1],3)









    b_e = b_e.detach().numpy()
    E_predict = (E_predict/N_atoms).detach().numpy()
    v_b_e = v_b_e.detach().numpy()
    v_E_predict = (v_E_predict/v_N_atoms).detach().numpy()



    if is_force:
        b_f = b_f.view(-1).detach().numpy()
        F_predict = F_predict.view(-1).detach().numpy()
        v_b_f = v_b_f.view(-1).detach().numpy()
        v_F_predict = v_F_predict.view(-1).detach().numpy()

    E_MAE = np.mean(abs(b_e - E_predict))
    v_E_MAE = np.mean(abs(v_b_e - v_E_predict)) 
    if is_force:
        F_MAE = np.sum(abs(b_f - F_predict)) / (3 * np.sum(N_atoms.detach().numpy()))
        v_F_MAE = np.sum(abs(v_b_f - v_F_predict)) / (3 * np.sum(v_N_atoms.detach().numpy()))

    # plt.clf()
    # plt.subplot(1,2,1)
    # plt.scatter(b_e,E_predict)
    # plt.plot(b_e,b_e)
    # plt.xlabel('EMT predicted energy (eV/atom)')
    # plt.ylabel('NN predicted energy (eV/atom)')
    # plt.subplot(1,2,2)
    # plt.scatter(v_b_e,v_E_predict)
    # plt.plot(v_b_e,v_b_e)
    # plt.xlabel('EMT predicted energy (eV/atom)')
    # plt.ylabel('NN predicted energy (eV/atom)')
    # plt.savefig('E_parity.png')


    # plt.clf()
    # plt.subplot(1,2,1)
    # plt.scatter(b_f,F_predict)
    # plt.plot(b_f,b_f)
    # plt.xlabel('EMT predicted forces (eV/A)')
    # plt.ylabel('NN predicted forces (eV/A)')
    # plt.subplot(1,2,2)
    # plt.scatter(v_b_f,v_F_predict)
    # plt.plot(v_b_f,v_b_f)
    # plt.xlabel('EMT predicted forces (eV/A)')
    # plt.ylabel('NN predicted forces (eV/A)')
    # plt.savefig('F_parity.png')

    plt.clf()
    matplotlib.rc('font',size=15)
    plt.figure(figsize=(10,15))
    ax = plt.subplot(2,1,1,aspect = 'equal')
    plt.scatter(b_e,E_predict)
    #plt.scatter(v_b_e,v_E_predict)
    x0,x1 = plt.xlim()

    plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
    plt.xlabel('DFT predicted energy (eV/atom)')
    plt.ylabel('NN predicted energy (eV/atom)')
    plt.title('Training Set')
    y0,y1 = plt.ylim()
    plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(E_MAE))

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    ax = plt.subplot(2,1,2, aspect = 'equal')
    
    plt.scatter(v_b_e,v_E_predict)
    x0,x1 = plt.xlim()
    
    plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
    plt.xlabel('DFT predicted energy (eV/atom)')
    plt.ylabel('NN predicted energy (eV/atom)')
    plt.title('Test Set')
    y0,y1 = plt.ylim()
    plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(v_E_MAE))
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.savefig('E_parity.png')


    if is_force:
        plt.clf()
        matplotlib.rc('font',size=15)
        plt.figure(figsize=(10,15))
        ax = plt.subplot(2,1,1,aspect = 'equal')
        plt.scatter(b_f,F_predict)
        x0,x1 = plt.xlim()

        plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
        plt.xlabel('DFT predicted forces (eV/A)')
        plt.ylabel('NN predicted forces (eV/A)')
        plt.title('Training Set')
        y0,y1 = plt.ylim()
        plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(F_MAE))

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        
        ax = plt.subplot(2,1,2, aspect = 'equal')
        
        plt.scatter(v_b_f,v_F_predict)
        x0,x1 = plt.xlim()
        
        plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
        plt.xlabel('DFT predicted forces (eV/A)')
        plt.ylabel('NN predicted forces (eV/A)')
        plt.title('Test Set')
        y0,y1 = plt.ylim()
        plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(v_F_MAE))
        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        plt.savefig('F_parity.png')



    # Adjust the energy prediction to center to 0
    v_b_e_adj = v_b_e - np.mean(v_b_e)
    v_E_predict_adj = v_E_predict - np.mean(v_E_predict)
    v_E_MAE_adj = np.mean(abs(v_b_e_adj - v_E_predict_adj))

    plt.clf()
    plt.figure(figsize=(10,10))
    plt.scatter(v_b_e_adj,v_E_predict_adj)
    x0,x1 = plt.xlim()
    
    plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
    plt.xlabel('DFT predicted energy (eV/atom)')
    plt.ylabel('NN predicted energy (eV/atom)')
    plt.title('Adjusted Prediction Set')
    y0,y1 = plt.ylim()
    plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(v_E_MAE_adj))
    plt.savefig('E_parity_adj.png')
