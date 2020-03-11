import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import torch
from ContextManager import cd

from Batch import batch_pad
import torch
import numpy as np
from torch.autograd import grad


Name = 'combined-1'

element_scaling = False

Element = ['Li', 'Si', 'Ni', 'Cu', 'Ge', 'Mo']


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
    b_fp.requires_grad = True
    sb_fp = (b_fp - gmin) / (gmax - gmin)
    N_atoms = batch_info['N_atoms'].view(-1)
    b_e = batch_info['b_e'].view(-1)
    b_f = batch_info['b_f'] 

    v_ids = np.array(list(val_dict.keys()))
    v_batch_info = batch_pad(val_dict,v_ids)
    v_b_fp = v_batch_info['b_fp']
    v_b_e_mask = v_batch_info['b_e_mask']
    v_b_fp.requires_grad = True
    v_sb_fp = (v_b_fp - gmin) / (gmax - gmin)
    v_N_atoms = v_batch_info['N_atoms'].view(-1)
    v_b_e = v_batch_info['b_e'].view(-1)
    v_b_f = v_batch_info['b_f'] 

    Atomic_Es = model(sb_fp)
    E_predict = torch.sum(Atomic_Es * b_e_mask, dim = [1,2])
    if element_scaling:
        b_emin = torch.mv(b_e_mask[:,0,:],emin)
        b_emax = torch.mv(b_e_mask[:,0,:],emax)
        E_predict = E_predict * (b_emax - b_emin) + b_emin * N_atoms
    else:
        E_predict = E_predict * (emax-emin) + emin * N_atoms
    b_dEdfp = grad(E_predict,
                   b_fp,
                   grad_outputs=torch.ones_like(E_predict),
                   create_graph = True,
                   retain_graph = True)[0].view(b_fp.shape[0],1,b_fp.shape[1]*b_fp.shape[2])
    b_dfpdX = batch_info['b_dfpdX'].view(b_fp.shape[0],b_fp.shape[1]*b_fp.shape[2],b_fp.shape[1]*3)
    F_predict = - torch.bmm(b_dEdfp,b_dfpdX).view(b_fp.shape[0],b_fp.shape[1],3)

    
    v_Atomic_Es = model(v_sb_fp)
    v_E_predict = torch.sum(v_Atomic_Es * v_b_e_mask, dim = [1,2])
    if element_scaling:
        v_b_emin = torch.mv(v_b_e_mask[:,0,:],emin)
        v_b_emax = torch.mv(v_b_e_mask[:,0,:],emax)
        v_E_predict = v_E_predict * (v_b_emax - v_b_emin) + v_b_emin * v_N_atoms
    else:
        v_E_predict = v_E_predict * (emax-emin) + emin * v_N_atoms
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




    b_f_all = b_f.view(-1).detach().numpy()
    F_predict_all = F_predict.view(-1).detach().numpy()
    v_b_f_all = v_b_f.view(-1).detach().numpy()
    v_F_predict_all = v_F_predict.view(-1).detach().numpy()

    E_MAE = np.mean(abs(b_e - E_predict))
    v_E_MAE = np.mean(abs(v_b_e - v_E_predict)) 
    F_MAE = np.sum(abs(b_f_all - F_predict_all)) / (3 * np.sum(N_atoms.detach().numpy()))
    v_F_MAE = np.sum(abs(v_b_f_all - v_F_predict_all)) / (3 * np.sum(v_N_atoms.detach().numpy()))


    plt.clf()
    matplotlib.rc('font',size=15)
    plt.figure(figsize=(20,20))
    ax = plt.subplot(1,2,1,aspect = 'equal')
    plt.scatter(b_e,E_predict)
    x0,x1 = plt.xlim()

    plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
    plt.xlabel('DFT predicted energy (eV/atom)')
    plt.ylabel('NN predicted energy (eV/atom)')
    plt.title('Training Set')
    y0,y1 = plt.ylim()
    plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(E_MAE))

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    ax = plt.subplot(1,2,2, aspect = 'equal')
    
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



    plt.clf()
    matplotlib.rc('font',size=15)
    plt.figure(figsize=(20,20))
    ax = plt.subplot(1,2,1,aspect = 'equal')
    plt.scatter(b_f_all,F_predict_all)
    x0,x1 = plt.xlim()

    plt.plot([x0,x1],[x0,x1],ls='--',color='grey')
    plt.xlabel('DFT predicted forces (eV/A)')
    plt.ylabel('NN predicted forces (eV/A)')
    plt.title('Training Set')
    y0,y1 = plt.ylim()
    plt.text(x0+0.25*(x1-x0),y1-0.25*(y1-y0),'MAE = {:.4f}'.format(F_MAE))

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    ax = plt.subplot(1,2,2, aspect = 'equal')
    
    plt.scatter(v_b_f_all,v_F_predict_all)
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





    b_f = b_f.detach().numpy()
    F_predict = F_predict.detach().numpy()
    v_b_f = v_b_f.detach().numpy()
    v_F_predict = v_F_predict.detach().numpy()

    E_predict_list = [np.array(())]*len(Element)
    E_list = [np.array(())]*len(Element)
    F_predict_list = [np.array(())]*len(Element)
    F_list = [np.array(())]*len(Element)
    N_atoms_list = [np.array(())]*len(Element)
    
    for n, e_ma in enumerate(b_e_mask):
        e_m = e_ma[0]
        i = (e_m == 1).nonzero().detach().numpy()[0][0]
        E_predict_list[i] = np.append(E_predict_list[i], E_predict[n])
        E_list[i] = np.append(E_list[i], b_e[n])
        F_predict_list[i] = np.append(F_predict_list[i], F_predict[n])
        F_list[i] = np.append(F_list[i], b_f[n])
        N_atoms_list[i] = np.append(N_atoms_list[i],N_atoms.detach().numpy()[n])

    v_E_predict_list = [np.array(())]*len(Element)
    v_E_list = [np.array(())]*len(Element)
    v_F_predict_list = [np.array(())]*len(Element)
    v_F_list = [np.array(())]*len(Element)
    v_N_atoms_list = [np.array(())]*len(Element)

    for n, e_ma in enumerate(v_b_e_mask):
        e_m = e_ma[0]
        i = (e_m == 1).nonzero().detach().numpy()[0][0]
        v_E_predict_list[i] = np.append(v_E_predict_list[i], v_E_predict[n])
        v_E_list[i] = np.append(v_E_list[i], v_b_e[n])
        v_F_predict_list[i] = np.append(v_F_predict_list[i], v_F_predict[n])
        v_F_list[i] = np.append(v_F_list[i], v_b_f[n])
        v_N_atoms_list[i] = np.append(v_N_atoms_list[i],v_N_atoms.detach().numpy()[n])

    with open(f'metrics', 'w') as f:
        for i, ele in enumerate(Element):
            f.write(ele+f': E_MAE = {np.mean(abs(E_list[i] - E_predict_list[i]))}\n')
        for i, ele in enumerate(Element):
            f.write(ele+f': v_E_MAE = {np.mean(abs(v_E_list[i] - v_E_predict_list[i]))}\n')
        for i, ele in enumerate(Element):
            f.write(ele+f': F_MAE = {np.sum(abs(F_list[i] - F_predict_list[i])) / (3 * np.sum(N_atoms_list[i]))}\n')
        for i, ele in enumerate(Element):
            f.write(ele+f': v_F_MAE = {np.sum(abs(v_F_list[i] - v_F_predict_list[i])) / (3 * np.sum(v_N_atoms_list[i]))}\n')   
