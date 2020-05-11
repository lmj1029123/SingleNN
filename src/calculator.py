import sys

sys.path.append("./SimpleNN")
import torch
from fp_calculator import set_sym, calculate_fp
from ase.data import atomic_numbers
from torch.autograd import grad
import pickle

def calculate(model_path, atoms, cal_list = None):
    model = torch.load(model_path+'/best_model')
    sym_params = pickle.load(open( model_path+"/sym_params.sav", "rb" ))

    [Gs, cutoff, g2_etas, g2_Rses, g4_etas, g4_zetas, g4_lambdas, elements, weights, element_energy]=sym_params

    params_set = set_sym(elements, Gs, cutoff,
                        g2_etas=g2_etas, g2_Rses=g2_Rses,
                        g4_etas=g4_etas, g4_zetas = g4_zetas,
                        g4_lambdas= g4_lambdas, weights=weights)
    if cal_list is None:
        N_atoms = len(atoms)
    else:
        N_atoms = len(cal_list)
    nelem = len(elements)
    N_sym = params_set[elements[0]]['num']
    data = calculate_fp(atoms, elements, params_set, cal_list = cal_list)
    fps = data['x']
    dfpdXs = data['dx']
    
    fp = torch.zeros((N_atoms,N_sym))
    dfpdX = torch.zeros((N_atoms, N_sym, N_atoms, 3))
    
    elements_num = torch.tensor([atomic_numbers[ele] for ele in elements])
    atom_idx = data['atom_idx'] - 1
    
    a_num = elements_num[atom_idx]
    atom_numbers = a_num.repeat_interleave(nelem).view(len(a_num),nelem)
    
    # change to float for pytorch to be able to run without error
    if cal_list is not None:
        e_mask = (atom_numbers == elements_num).float()[cal_list]
        atom_idx = atom_idx[cal_list]
    else:
        e_mask = (atom_numbers == elements_num).float()
    fp_track = [0]*nelem
    element_energy = torch.sum(element_energy * e_mask)

    for i,idx in enumerate(atom_idx):
        ele = elements[idx]
        fp[i,:] = torch.tensor(fps[ele][fp_track[idx],:]).float()
        dfpdX[i,:,:,:] = torch.tensor(dfpdXs[ele][fp_track[idx],:,:,:]).float()
        fp_track[idx] += 1
    fp.requires_grad = True
    scaling = model.scaling
    gmin = scaling['gmin']
    gmax = scaling['gmax']
    emin = scaling['emin']
    emax = scaling['emax']
    sfp = (fp - gmin) / (gmax - gmin)
    Atomic_Es = model(sfp)
    E_predict = torch.sum(torch.sum(Atomic_Es * e_mask,
                                    dim = 1)*(emax-emin)+emin,dim=0)
    dEdfp = grad(E_predict,
                 fp,
                 grad_outputs=torch.ones_like(E_predict),
                 create_graph = True,
                 retain_graph = True)[0].view(1,fp.shape[0]*fp.shape[1])
    dfpdX = dfpdX.view(fp.shape[0]*fp.shape[1],fp.shape[0]*3)
    F_predict = -torch.mm(dEdfp,dfpdX).view(fp.shape[0],3)
    return E_predict + element_energy, F_predict
