import torch
import os
from src.dataset import Multimodal_Datasets, MMDataset


def get_data(args, dataset, split='train'):
    print('dataset name:', dataset)
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'

    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")

        if args.use_bert:
            data = MMDataset(args.data_path, use_bert=True, need_norm=False, train_mode=args.train_mode,
                             data=dataset, split=split, if_align=args.aligned)
        else:
            data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
            torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'all_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'all_models/{name}.pt')
    return model
