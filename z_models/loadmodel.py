import torch
from .BiSeNet_model import BiSeNet


def show_paramsnumber(net, netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters / 1e6, 2)
    print(netname + ' parameters: ' + str(parameters) + 'M')


def bisenet(opt, type='roi'):
    '''
    type: roi or mosaic
    '''
    net = BiSeNet(num_classes=1, context_path='resnet18', train_flag=False)
    show_paramsnumber(net, 'segment')
    if type == 'roi':
        net.load_state_dict(torch.load(opt.model_path))
    elif type == 'mosaic':
        net.load_state_dict(torch.load(opt.mosaic_position_model_path))
    net.eval()
    if opt.use_gpu:
        net.cuda()
    return net
