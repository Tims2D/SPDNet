from data_provider.data_loader import Dataset_Custom, ML_DataLoader
from torch.utils.data import DataLoader

data_dict = {'ETTm1': Dataset_Custom, 
             'aggregated_data':Dataset_Custom,
             'Load1': Dataset_Custom,
             'ml': ML_DataLoader}

def data_provider(args, flag):
    
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    args.flag = flag
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(configs=args)  # Pass the entire args object as configs
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=(flag != 'test'),  # Shuffle during training and validation, not during testing
        num_workers=args.num_workers,
        drop_last=True
    )
    
    return data_set, data_loader
