
from model import *
from model_ct_net import *


def init_model_MK2(dataset, model, device, batch_size=128 ,sd_path=None, **config):

    # dataset_list = ["A","B","C"]
    # dataset = dataset_list[dataset_id]
    print(f"init_model: dataset '{dataset}'")
    match dataset:
        case "A":
            n_class = 2
            sampling_point = 140
            h = 52
            ct_f = int(5600)
        case "B":
            n_class = 2
            sampling_point = 200
            h = 36
            ct_f = int(8000)
        case "C":
            n_class = 3
            sampling_point = 256
            h = 20
            ct_f = int(sampling_point*h*2)
    
    match model:
        case 'fNIRS-T':
            net = fNIRS_T(n_class=n_class, sampling_point=sampling_point, h=h, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
        
        case 'fNIRS-PreT':
            net = fNIRS_PreT(n_class=n_class, sampling_point=sampling_point, h=h, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
    

        case 'fNIRS_TTT_M':
            net = fNIRS_TTT_M(n_class=n_class, sampling_point=sampling_point, 
                             dim=64, depth=6, heads=8, mlp_dim=64, device=device, 
                             batch_size=batch_size, dataset=dataset).to(device)
        
            
        case 'fNIRS_TTT_L':
            net = fNIRS_TTT_L(n_class=n_class, sampling_point=sampling_point,
                             dim=64, depth=6, heads=8, mlp_dim=64, device=device,
                             batch_size=batch_size, dataset=dataset).to(device)

        case 'CT-Net':
            # net = CTNet_old(heads=4, emb_size=40,depth=6, eeg1_f1=20, 
            #             eeg1_D=2,eeg1_kernel_size=16, eeg1_pooling_size1=2, eeg1_pooling_size2=2,
            #             eeg1_dropout_rate=0.1,eeg1_number_channel=h,flatten_eeg1=ct_f,n_class=n_class, device=device).to(device) # A 80 B 120 C 160
            net = CTNet(heads=4, emb_size=40,depth=6, eeg1_f1=20, 
                        eeg1_D=2,eeg1_kernel_size=32, eeg1_pooling_size1=2, eeg1_pooling_size2=2,
                        eeg1_dropout_rate=0.1,eeg1_number_channel=h,flatten_eeg1=ct_f,n_class=n_class, device=device).to(device) # A 80 B 120 C 160

        case _:
            print("Unknown model in model_init!")
            breakpoint()

    print(f"init {net}!")
    
    if sd_path:
        sd = torch.load(sd, map_location=device)
        net.load_state_dict(sd)

    return net                

 