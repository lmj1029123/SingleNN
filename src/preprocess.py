import sys

sys.path.append("./SimpleNN")
import torch
from ase.db import connect
import numpy as np
from ase import Atom
from ase.data import atomic_numbers
from fp_calculator import calculate_fp




def snn2sav(db,directory,elements,params_set, element_energy=None):
    """The energy is eV/atom
    The dfpdX is /A"""
    
    nelem = len(elements)
    N_sym = params_set[elements[0]]['num']
    data_dict = {}
    i1 = 1
    for row in db.select():
        atoms = row.toatoms()
        N_atoms= len(atoms)
        energy = torch.tensor(row.energy).float()
        forces = torch.tensor(row.forces).float()
        data = calculate_fp(atoms, elements, params_set)
        fps = data['x']
        dfpdXs = data['dx']
       

        fp = torch.zeros((N_atoms,N_sym))
        dfpdX = torch.zeros((N_atoms, N_sym, N_atoms, 3))
        
        elements_num = torch.tensor([atomic_numbers[ele] for ele in elements])
        atom_idx = data['atom_idx'] - 1
        
        a_num = elements_num[atom_idx]
        atom_numbers = a_num.repeat_interleave(nelem).view(len(a_num),nelem)

        # change to float for pytorch to be able to run without error
        e_mask = (atom_numbers == elements_num).float()
       # print(e_mask)

        fp_track = [0]*nelem
        for i,idx in enumerate(atom_idx):
            ele = elements[idx]
            fp[i,:] = torch.tensor(fps[ele][fp_track[idx],:]).float()
            dfpdX[i,:,:,:] = torch.tensor(dfpdXs[ele][fp_track[idx],:,:,:]).float()
            fp_track[idx] += 1
            
                
           
        if element_energy is not None:
            energy -= torch.sum(e_mask * element_energy.float())

        energy = energy/N_atoms
            
        data_dict[i1]={'fp':fp,'dfpdX':dfpdX,'e_mask':e_mask,'e':energy,'f':forces}
        i1 += 1

    print('preprocess done')
    return data_dict

def train_test_split(data_dict, train_percent, seed=42):
    ids = np.array(list(data_dict.keys()))
    np.random.seed(seed)
    np.random.shuffle(ids)

    N_total = len(data_dict)
    N_train = int(N_total * train_percent)
    tr_ids = ids[:N_train]
    tt_ids = ids[N_train:]
    train_dict = dict((i, data_dict[i]) for i in tr_ids)
    test_dict = dict((i, data_dict[i]) for i in tt_ids)
    torch.save(test_dict, 'test.sav')
    print('train_test_split done')
    return train_dict


def train_val_split(data_dict, train_percent, seed=42):
    ids = np.array(list(data_dict.keys()))
    np.random.seed(seed)
    np.random.shuffle(ids)

    N_total = len(data_dict)
    N_train = int(N_total * train_percent)
    t_ids = ids[:N_train]
    v_ids = ids[N_train:]
    train_dict = dict((i, data_dict[i]) for i in t_ids)
    val_dict = dict((i, data_dict[i]) for i in v_ids)
    torch.save(train_dict, 'final_train.sav')
    torch.save(val_dict, 'final_val.sav')
    print('final train_val_split done')
    return

def CV(k_fold, seed=42):
    """
    This function implments the cross-validation split.
    The seed is for reproducibility.
    """
    data_dict = torch.load('train.sav')
    ids = np.array(list(data_dict.keys()))
    np.random.seed(seed)
    np.random.shuffle(ids)
    ids = np.array_split(ids,k_fold)

    TRAIN_dict = []
    VAL_dict = []
    for k in range(k_fold):
        v_ids = ids[k]
        t_ids = np.concatenate([ids[j] for j in range(k_fold) if j!=k])
        train_dict = dict((i, data_dict[i]) for i in t_ids)
        val_dict = dict((i, data_dict[i]) for i in v_ids)
        TRAIN_dict += [train_dict]
        VAL_dict += [val_dict]
    torch.save(TRAIN_dict,'CV_train.sav')
    torch.save(VAL_dict,'CV_val.sav')

    return TRAIN_dict, VAL_dict

    

def get_scaling(train_dict, fp_scale_method, e_scale_method):
    train_ids = train_dict.keys()
    if fp_scale_method == 'min_max':
        all_fp = torch.tensor(())
        for ID in train_ids:
            fp = train_dict[ID]['fp']
            all_fp = torch.cat((all_fp, fp), dim=0)
        gmax = torch.max(all_fp, 0)[0]
        gmin = torch.min(all_fp, 0)[0]

    # This calculate the min and max energy/atom
    if e_scale_method == 'min_max':
        all_e = torch.tensor(())
        for ID in train_ids:
            e = train_dict[ID]['e'].view(1, 1)
            all_e = torch.cat((all_e, e), dim=0)
        emax = torch.max(all_e)
        emin = torch.min(all_e)
    return {'gmax':gmax, 'gmin':gmin, 'emax':emax, 'emin':emin}





