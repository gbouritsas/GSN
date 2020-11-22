from ogb.graphproppred import PygGraphPropPredDataset
import os

root_folder = '/vol/deform/gbouritsas/datasets/'

datasets = ['ogbg-molpcba', 'ogbg-molhiv', 'ogbg-ppa']

for name in datasets:
    dataset = PygGraphPropPredDataset(name=name, root=os.path.join(root_folder,'ogb','{}'.format(name)))
    split_idx = dataset.get_idx_split()
    for split_name in {'train', 'valid', 'test'}:
        idxs = split_idx[split_name]
        split_name = split_name if split_name is not 'valid' else 'val'
        save_folder = os.path.join(root_folder,'ogb','{}'.format(name),'10fold_idx')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder,'{}_idx-0.txt'.format(split_name)), 'w') as handle:
            for idx in idxs:
                handle.write('{}\n'.format(idx))