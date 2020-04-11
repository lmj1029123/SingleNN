import torch
import copy

def batch_pad(data_dict,ids):
    """
    This function combine the data into batches based on the ids provided.
    Also, it pad the images with shorter length with zeros. As a result, an
    image with small number of atoms might have very sparse matrices for fp
    and dfpdX.
    """
    batch_info = {}
    b_fp = torch.tensor(())
    b_dfpdX = torch.tensor(())
    b_e_mask = torch.tensor(())
    b_e = torch.tensor(())
    b_f = torch.tensor(())

    # Find the largest image in the batch
    N_max = 0
    all_atoms = torch.tensor(())
    for ID in ids:
        N_atoms, N_element = data_dict[ID]['e_mask'].shape
        all_atoms = torch.cat((all_atoms,torch.tensor(N_atoms).float().view(1,1)))
        if N_atoms > N_max:
            N_max = N_atoms
            N_sym = data_dict[ID]['fp'].shape[1]

    # Loop through the ids to batch the values
    for ID in ids:
        pad_fp = torch.zeros(N_max,N_sym)
        pad_dfpdX = torch.zeros(N_max,N_sym,N_max,3)
        pad_e_mask = torch.zeros(N_max,N_element)
        pad_f = torch.zeros(N_max,3)
        fp = data_dict[ID]['fp']
        dfpdX = data_dict[ID]['dfpdX']
        e_mask = data_dict[ID]['e_mask']
        pad_fp[:fp.shape[0],:fp.shape[1]] = fp
        pad_dfpdX[:dfpdX.shape[0],:dfpdX.shape[1],:dfpdX.shape[2],:] = dfpdX
        pad_e_mask[:e_mask.shape[0],:e_mask.shape[1]] = e_mask
        pad_f[:fp.shape[0],:] = data_dict[ID]['f'] 
        b_fp = torch.cat((b_fp,pad_fp))
        b_dfpdX = torch.cat((b_dfpdX,pad_dfpdX))
        b_e_mask = torch.cat((b_e_mask,pad_e_mask))
        b_e = torch.cat((b_e,data_dict[ID]['e'].view(1,1)),dim=0)
        b_f = torch.cat((b_f,pad_f))

    # Update the output dictionary
    batch_info.update({'N_atoms':all_atoms})
    batch_info.update({'b_fp':b_fp.view(len(ids),N_max,N_sym)})
    batch_info.update({'b_dfpdX':b_dfpdX.view(len(ids),N_max,N_sym,N_max,3)})
    batch_info.update({'b_e_mask':b_e_mask.view(len(ids),N_max,N_element)})
    batch_info.update({'b_e':b_e})
    batch_info.update({'b_f':b_f.view(len(ids),N_max,3)})
    return batch_info



