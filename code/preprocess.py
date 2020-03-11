import torch

from ase.db import connect
import numpy as np
from ase import Atom



def train_test_split(train_percent, seed=42):
    data_dict = torch.load('data.sav')
    ids = np.array(list(data_dict.keys()))
    np.random.seed(seed)
    np.random.shuffle(ids)

    N_total = len(data_dict)
    N_train = int(N_total * train_percent)
    tr_ids = ids[:N_train]
    tt_ids = ids[N_train:]
    train_dict = dict((i, data_dict[i]) for i in tr_ids)
    test_dict = dict((i, data_dict[i]) for i in tt_ids)
    torch.save(train_dict, 'train.sav')
    torch.save(test_dict, 'test.sav')
    print('train_test_split done')


def train_val_split(train_percent, seed=42):
    data_dict = torch.load('train.sav')
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




def get_element_scaling(train_dict, fp_scale_method, e_scale_method):
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
            e_mask = train_dict[ID]['e_mask']
            e = train_dict[ID]['e'].view(1, 1) * e_mask
            all_e = torch.cat((all_e, e), dim=0)
        nelement = all_e.shape[1]
        emax = torch.zeros((nelement))
        emin = torch.zeros((nelement))
        for i in range(nelement):
            ae = all_e[:,i]
            ae = ae[ae.nonzero()]
            emax[i] = torch.max(ae)
            emin[i] = torch.min(ae)
        print(emax, emin,emax.shape,emin.shape)
        #ds
    return {'gmax':gmax, 'gmin':gmin, 'emax':emax, 'emin':emin}


def get_element_energy(dbfile,element):
    db = connect(dbfile)
    element_energy = torch.zeros_like(element)
    for row in db.select():
        energy = row.energy
        atoms = row.toatoms()
        number = atoms[0].number
        N = len(atoms)
        energy = energy/2
        index = (element==number).nonzero()[0]
        element_energy[index] = energy

    return element_energy


def get_symfunctions(symfunc,element):
    """Helper function to generate the symfunction configuration for all elements""" 
    symfunctions = {}
    for ele in element:
        symbol = Atom(ele).symbol
        symfunctions[symbol] = symfunc
    return symfunctions 
