"""SingleNN potential."""

from math import sqrt, exp, log

import numpy as np

from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)



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

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'numbers' in system_changes:
            self.initialize(self.atoms)

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell

        self.nl.update(self.atoms)

        self.energy = 0.0
        self.energies[:] = 0
        self.sigma1[:] = 0.0
        self.forces[:] = 0.0
        self.stress[:] = 0.0

        natoms = len(self.atoms)

        for a1 in range(natoms):
            Z1 = numbers[a1]
            p1 = self.par[Z1]
            ksi = self.ksi[Z1]
            neighbors, offsets = self.nl.get_neighbors(a1)
            offsets = np.dot(offsets, cell)
            for a2, offset in zip(neighbors, offsets):
                d = positions[a2] + offset - positions[a1]
                r = sqrt(np.dot(d, d))
                if r < self.rc_list:
                    Z2 = numbers[a2]
                    p2 = self.par[Z2]
                    self.interact1(a1, a2, d, r, p1, p2, ksi[Z2])

        for a in range(natoms):
            Z = numbers[a]
            p = self.par[Z]
            try:
                ds = -log(self.sigma1[a] / 12) / (beta * p['eta2'])
            except (OverflowError, ValueError):
                self.deds[a] = 0.0
                self.energy -= p['E0']
                self.energies[a] -= p['E0']
                continue
            x = p['lambda'] * ds
            y = exp(-x)
            z = 6 * p['V0'] * exp(-p['kappa'] * ds)
            self.deds[a] = ((x * y * p['E0'] * p['lambda'] + p['kappa'] * z) /
                            (self.sigma1[a] * beta * p['eta2']))
            E = p['E0'] * ((1 + x) * y - 1) + z
            self.energy += E
            self.energies[a] += E

        for a1 in range(natoms):
            Z1 = numbers[a1]
            p1 = self.par[Z1]
            ksi = self.ksi[Z1]
            neighbors, offsets = self.nl.get_neighbors(a1)
            offsets = np.dot(offsets, cell)
            for a2, offset in zip(neighbors, offsets):
                d = positions[a2] + offset - positions[a1]
                r = sqrt(np.dot(d, d))
                if r < self.rc_list:
                    Z2 = numbers[a2]
                    p2 = self.par[Z2]
                    self.interact2(a1, a2, d, r, p1, p2, ksi[Z2])

        self.results['energy'] = self.energy
        self.results['energies'] = self.energies
        self.results['free_energy'] = self.energy
        self.results['forces'] = self.forces

        if 'stress' in properties:
            raise PropertyNotImplementedError

def calculate(self, atoms=None, properties=['energy'],
                system_changes=all_changes,cal_list = None):
    model = torch.load(self.model_path+'/best_model')
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
    
    self.forces = F_predict
    self.energy = E_predict + element_energy
    self.results['energy'] = self.energy
    self.results['free_energy'] = self.energy
    self.results['forces'] = self.forces

    if 'stress' in properties:
            raise PropertyNotImplementedError