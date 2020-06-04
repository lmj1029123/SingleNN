import sys

sys.path.append("./SimpleNN")
"""SingleNN potential."""
from fp_calculator import set_sym, calculate_fp
from math import sqrt, exp, log

import numpy as np
import torch
import pickle
from torch.autograd import grad
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)
import os


class SingleNN(Calculator):
    """Python implementation of the Effective Medium Potential.

    Supports the following standard EMT metals:
    Al, Cu, Ag, Au, Ni, Pd and Pt.

    In addition, the following elements are supported.
    They are NOT well described by EMT, and the parameters
    are not for any serious use:
    H, C, N, O

    The potential takes a single argument, ``asap_cutoff``
    (default: False).  If set to True, the cutoff mimics
    how Asap does it; most importantly the global cutoff
    is chosen from the largest atom present in the simulation,
    if False it is chosen from the largest atom in the parameter
    table.  True gives the behaviour of the Asap code and
    older EMT implementations, although the results are not
    bitwise identical.
    """
    implemented_properties = ['energy', 'energies', 'forces']

    def __init__(self, model_path,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.model_path = model_path

    def initialize(self, atoms):
        self.numbers = atoms.get_atomic_numbers()
        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))


    def calculate(self, atoms=None, properties=['energy'],system_changes=all_changes,cal_list = None):
        if os.path.exists(self.model_path+'/best_model'):
            model = torch.load(self.model_path+'/best_model')
            ensemble_training = False
        else:
            ensemble_training = True 
            models = []
            ensemble = 0
            end = False
            while end is False:
                if os.path.exists(self.model_path+f'/best_model-{ensemble}'):
                    models += [torch.load(self.model_path+f'/best_model-{ensemble}')]
                    ensemble += 1
                else:
                    end = True



        sym_params = pickle.load(open(self.model_path+"/sym_params.sav", "rb" ))
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
        if element_energy is not None:
            element_energy = torch.sum(element_energy * e_mask)

        for i,idx in enumerate(atom_idx):
            ele = elements[idx]
            fp[i,:] = torch.tensor(fps[ele][fp_track[idx],:]).float()
            dfpdX[i,:,:,:] = torch.tensor(dfpdXs[ele][fp_track[idx],:,:,:]).float()
            fp_track[idx] += 1
        fp.requires_grad = True

        if ensemble_training:
            scaling = models[0].scaling
        else:
            scaling = model.scaling

        gmin = scaling['gmin']
        gmax = scaling['gmax']
        emin = scaling['emin']
        emax = scaling['emax']
        sfp = (fp - gmin) / (gmax - gmin)

        if ensemble_training:
            all_energy = []
            all_forces = []
            for model in models:
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
                forces = F_predict.data.numpy()
                if element_energy is not None:
                    energy = (E_predict + element_energy).data.numpy()
                else:
                    energy = E_predict.data.numpy()
                all_energy += [energy]
                all_forces += [forces]
            all_energy = np.array(all_energy)
            all_forces = np.array(all_forces)
            ensemble_energy = np.mean(all_energy)
            energy_std = np.std(all_energy)
            ensemble_forces = np.mean(all_forces, axis=0)
            forces_std = np.std(all_forces, axis=0)
            self.energy = ensemble_energy
            self.forces = ensemble_forces
            self.results['energy'] = self.energy
            self.results['free_energy'] = self.energy
            self.results['forces'] = self.forces
            self.results['energy_std'] = energy_std
            self.results['forces_std'] = forces_std
        else:
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
            self.forces = F_predict.data.numpy()
            if element_energy is not None:
                self.energy = (E_predict + element_energy).data.numpy()
            else:
                self.energy = E_predict.data.numpy()
            self.results['energy'] = self.energy
            self.results['free_energy'] = self.energy
            self.results['forces'] = self.forces

        if 'stress' in properties:
            raise PropertyNotImplementedError