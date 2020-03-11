import torch




SEED = [1]
Name = 'combined'
FN =['test','train','final_train','final_val']
Element = ['Li','Si','Ni','Cu','Ge','Mo']

for seed in SEED:
    for filename in FN:
        n_dict = {}
        i = 0
        for ele in Element:
            data_dict = torch.load(f'./{ele}-{seed}/{filename}.sav')
            for key in data_dict.keys():
                data = data_dict[key]
                i += 1
                n_dict[i] = data
    
        torch.save(n_dict,f'./{Name}-{seed}/{filename}.sav')



