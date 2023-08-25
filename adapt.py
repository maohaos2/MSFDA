import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from data_list import ImageList, ImageList_idx
from loss import mutual_info_loss, prototype_weights
from torch.utils.data import DataLoader
import random

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def data_load(args):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets['target_'] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders['target_'] = DataLoader(dsets['target_'], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

# Mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    # Returns mixed inputs, pairs of targets, and lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


"""
Get the predictions from pretrained source models for initialization
"""
def initialize(loader, netF, netB, netC):
    flag = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            targets = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            outputs = netC(feas)
            if flag:
                all_logits = outputs.float().cpu()
                all_targets = targets.float().cpu()
                flag = False
            else:
                all_logits = torch.cat((all_logits, outputs.float().cpu()), 0)
                all_targets = torch.cat((all_targets, targets.float().cpu()), 0)
    _, predict = torch.max(all_logits, 1)
    predict = torch.squeeze(predict).float()
    return predict, all_logits, all_targets

"""
Prototype based pseudo label denoising
"""
def prototype(loader, netF, netB, bandwidth, pseudo_label):
    mask = pseudo_label < 0
    flag = True
    with torch.no_grad():
        netF.eval()
        netB.eval()
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            if flag:
                all_feas = feas.float().cpu()
                flag = False
            else:
                all_feas = torch.cat((all_feas, feas.float().cpu()), 0)
    for _ in range(args.prototype_itr):
        center = torch.zeros((args.class_num, args.bottleneck))
        n = all_feas.size()[0]
        counter = torch.zeros(args.class_num)
        for i in range(n):
            if pseudo_label[i] >= 0:
                index = int(pseudo_label[i])
                center[index] += all_feas[i]
                counter[index] += 1
        for i in range(args.class_num):
            if counter[i] > 0:
                center[i] /= counter[i]
        prototype_w = prototype_weights(all_feas, center, bandwidth)
        _, pseudo_label = torch.max(prototype_w, 1)
        pseudo_label[mask] = -1
    return prototype_w

"""
Domain Aggregation
"""
def domain_aggregation(predictions, netW, optimizer_w, w_epoch):
    netW.train()
    for t in range(w_epoch):
        predictions_weighted = netW(torch.stack(predictions, dim=-1).cuda())
        loss = mutual_info_loss(predictions_weighted)
        optimizer_w.zero_grad()
        loss.backward()
        optimizer_w.step()
    netW.eval()
    predictions_weighted = netW(torch.stack(predictions, dim=-1).cuda()).cpu()
    domain_weights = torch.abs(netW.domain_weights).cpu()
    domain_weights = domain_weights / (torch.sum(domain_weights) + 1e-5)
    return domain_weights, predictions_weighted, netW

"""
Selective pseudo labeling
"""
def update_pseudo_label(domain_weights, prototype_list, predictions_weighted, all_targets, num_selected=None, num_queried=None):
    n = predictions_weighted.size()[0]
    predictions_ensemble =predictions_weighted* torch.sum(torch.stack([domain_weights[i]*prototype_list[i] for i in range(len(args.src))]),0)
    confidence, pseudo_label_raw = torch.max(predictions_ensemble, 1)
    pseudo_label = torch.zeros(n) - 1
    if num_queried==None:
        confidence_mean = confidence.mean().item()
        index = confidence >= args.lambda_confidence * confidence_mean
    else:
        index = confidence.argsort()[-(num_queried + num_selected):]
    if args.upper_bound== 'True':
        index = pseudo_label_raw == all_targets
    pseudo_label[index] = pseudo_label_raw[index].float()
    return pseudo_label, pseudo_label_raw

"""
Retrain the multiple source models to adapt to target domain
"""
def retrain_model(netF_list, netB_list, netC_list, netD, optim_group, optim_domain, pseudo_label):
    for i in range(len(args.src)):
        netF_list[i].train()
        netB_list[i].train()
    netD.train()
    # Partition of target domain data
    dsets = {}
    dset_loaders = {}
    txt_tar = open(args.t_dset_path).readlines()
    N = len(txt_tar)
    mask = pseudo_label >= 0
    temp = pseudo_label[mask]
    txt_tar_c = []
    txt_tar_u = []
    for i in range(N):
        if mask[i]:
            txt_tar_c.append(txt_tar[i])
        else:
            txt_tar_u.append(txt_tar[i])
    dsets["D_l"] = ImageList_idx(txt_tar_c, pseudo_label[mask], transform=image_train())
    dset_loaders["D_l"] = DataLoader(dsets["D_l"], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['D_u'] = ImageList_idx(txt_tar_u, transform=image_train())
    dset_loaders["D_u"] = DataLoader(dsets["D_u"], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)

    max_iter = args.max_epoch * np.maximum(len(dset_loaders["D_u"]), len(dset_loaders["D_l"]))
    iter_num = 0
    criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    supervised_loss_avg = np.zeros(len(args.src))
    align_loss_avg = 0
    discriminator_loss_avg = 0
    MI_loss_avg = np.zeros(len(args.src))
    accuracy = np.zeros(len(args.src))
    data_num = 0
    while iter_num < max_iter:
        try:
            inputs_c, target_c, tar_idx_c = next(iter_test_c)
            inputs_u, _, tar_idx_u = next(iter_test_u)
        except:
            iter_test_c = iter(dset_loaders["D_l"])
            iter_test_u = iter(dset_loaders["D_u"])
            inputs_c, target_c, tar_idx_c = next(iter_test_c)
            inputs_u, _, tar_idx_u = next(iter_test_u)
        if inputs_c.size(0) <= 2 or inputs_u.size(0) <=2:
            continue

        iter_num += 1
        for i in range(len(args.src)):
            lr_scheduler(optim_group[i], iter_num=iter_num, max_iter=max_iter)
            netF_list[i].train()
            netB_list[i].train()
        netD.train()

        # Forward
        supervised_list = []
        MI_list = []
        feature_c_list = []
        feature_u_list = []
        inputs_c = inputs_c.cuda()
        target_c = target_c.cuda()
        inputs_u = inputs_u.cuda()
        data_num = data_num + target_c.size()[0]
        # Mixup
        inputs_mixup, targets_1, targets_2, lam = mixup_data(inputs_c, target_c.type(torch.LongTensor).cuda(), alpha=1.0, use_cuda=True)
        for i in range(len(args.src)):
            # mixup
            outputs_mixup = netC_list[i](netB_list[i](netF_list[i](inputs_mixup)))
            mixup_loss = mixup_criterion(criterion, outputs_mixup, targets_1, targets_2, lam)
            # data output
            feature_c = netB_list[i](netF_list[i](inputs_c))
            feature_c_list.append(feature_c)
            outputs_c = netC_list[i](feature_c)
            supervised_loss = criterion(outputs_c, target_c.type(torch.LongTensor).cuda())

            softmax_c = nn.Softmax(dim=1)(outputs_c)
            _, predict_c = torch.max(softmax_c, 1)
            accuracy[i] += torch.sum(predict_c == target_c).item()
            # inference on unlabeled data
            feature_u = netB_list[i](netF_list[i](inputs_u))
            feature_u_list.append(feature_u)
            # supervised training loss
            supervised_list.append(mixup_loss+supervised_loss)
            MI_list.append(mutual_info_loss(softmax_c))

        discrimate_c = netD(torch.cat(feature_c_list,dim=1))
        discrimate_u = netD(torch.cat(feature_u_list,dim=1))
        # Joint domain alignment loss
        domain_loss = (dom_criterion(discrimate_c,torch.zeros(inputs_c.size(0)).view(-1, 1).cuda()) + dom_criterion(discrimate_u, torch.ones(inputs_u.size(0)).view(-1, 1).cuda())) * 0.5
        domain_inverse_loss = (dom_criterion(discrimate_c,torch.ones(inputs_c.size(0)).view(-1, 1).cuda()) + dom_criterion(discrimate_u, torch.zeros(inputs_u.size(0)).view(-1, 1).cuda())) * 0.5

        # Backward
        with torch.autograd.set_detect_anomaly(True):
            for i in range(len(args.src)):
                supervised_loss_avg[i] += supervised_list[i].item()
                align_loss_avg += domain_loss.item()
                discriminator_loss_avg += domain_inverse_loss.item()
                MI_loss_avg[i] += MI_list[i].item()
                optim = optim_group[i]
                optim.zero_grad()
                total_loss = args.lambda_supervised * supervised_list[i] + args.lambda_align * domain_loss + args.lambda_MI * MI_list[i]
                total_loss.backward(retain_graph=True)
        optim2 = optim_domain
        optim2.zero_grad()
        discriminator_loss = args.lambda_align * domain_inverse_loss
        discriminator_loss.backward(retain_graph=True)
        for i in range(len(args.src)):
            optim = optim_group[i]
            optim.step()
        optim2.step()

        # Training statistics
        if iter_num == max_iter or iter_num % np.maximum(len(dset_loaders["D_u"]), len(dset_loaders["D_l"]))==0:
            num_iter = np.maximum(len(dset_loaders["D_u"]), len(dset_loaders["D_l"]))
            if args.verbose=='True':
                print('current iteration:', iter_num)
                print('Cross Entropy loss: {}, Joint Alignment Loss: {},MI Loss {}, Training Accuracy: {}'.format(supervised_loss_avg/num_iter, align_loss_avg/num_iter, MI_loss_avg/num_iter,
                                                                                                                    accuracy/data_num))
                print('Discriminator loss is{}'.format(discriminator_loss_avg/num_iter))
            supervised_loss_avg = np.zeros(len(args.src))
            align_loss_avg = 0
            discriminator_loss_avg = 0
            MI_loss_avg = np.zeros(len(args.src))
            accuracy = np.zeros(len(args.src))
            data_num = 0

    for i in range(len(args.src)):
        netF_list[i].eval()
        netB_list[i].eval()

    return netF_list, netB_list, netD

"""
Get prediction accuracy of individual model
"""
def cal_acc_model(loader, netF, netB, netC, all_targets):
    start_test = True
    with torch.no_grad():
        netF.eval()
        netB.eval()
        netC.eval()
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_logits = outputs.float().cpu()
                start_test = False
            else:
                all_logits = torch.cat((all_logits, outputs.float().cpu()), 0)

    _, predict = torch.max(all_logits, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_targets).item() / float(all_targets.size()[0])
    return accuracy, all_logits

"""
Get ensemble accuracy of multiple models with domain weights
"""
def ensemble(all_logits, all_targets, netW):
    netW.eval()
    all_output = netW(torch.stack(all_logits, dim=-1).cuda())
    _, predict = torch.max(all_output, 1)
    predict =  predict.cpu()
    accuracy = torch.sum(torch.squeeze(predict).float() == all_targets).item() / float(all_targets.size()[0])
    return accuracy

"""
Domain Adaptation framework
"""
def train_target(args):
    dset_loaders = data_load(args)
    # Set base network
    if args.net[0:3] == 'res':
        netF_list = [network.ResBase(res_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net[0:3] == 'vgg':
        netF_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]
    # Bottleneck of feature encoder
    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    # Classfier
    netC_list = [network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    # Domain discriminator
    netD = network.domain_discriminator(type=args.layer, bottleneck_dim=len(args.src)*args.bottleneck).cuda()
    netD.eval()
    # Domain Aggregation Module
    netW = network.domain_aggregation(len(args.src)).cuda()
    netW.eval()
    # Set optimizer
    optim_group = []
    for i in range(len(args.src)):
        param_group = []
        param_domain_group = []
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)
        optim_group.append(optimizer)

    param_domain = []
    for k, v in netD.named_parameters():
        param_domain += [{'params': v, 'lr': args.lr * args.lr_decay2}]
    optimizer_domain = optim.SGD(param_domain)
    optimizer_domain = op_copy(optimizer_domain)

    param_w = []
    for k, v in netW.named_parameters():
        param_w += [{'params': v, 'lr': 1e-1}]
    optimizer_w = optim.SGD(param_w)
    optimizer_w = op_copy(optimizer_w)

    for i in range(len(args.src)):
        netF_list[i] = nn.DataParallel(netF_list[i])
        netB_list[i] = nn.DataParallel(netB_list[i])
        netC_list[i] = nn.DataParallel(netC_list[i])
    netD = nn.DataParallel(netD)

    # Initialize the pseudo label and target data partition
    all_logits = []
    prototype_list = []
    for i in range(len(args.src)):
        netF_list[i].eval()
        netB_list[i].eval()
        netC_list[i].eval()
        predict_label, logits, all_targets = initialize(dset_loaders['test'], netF_list[i], netB_list[i], netC_list[i])
        prototype_w = prototype(dset_loaders['test'], netF_list[i], netB_list[i], args.lambda_bandwidth, predict_label)
        prototype_list.append(prototype_w)
        all_logits.append(logits)
        netF_list[i].train()
        netB_list[i].train()
    # Generate the pseudo label
    domain_weights, predictions_weighted, netW = domain_aggregation(all_logits, netW, optimizer_w, args.w_epoch)
    pseudo_label, pseudo_label_raw = update_pseudo_label(domain_weights, prototype_list, predictions_weighted, all_targets)

    mask = pseudo_label_raw == all_targets
    num_selected = pseudo_label[pseudo_label >= 0].size()[0]
    num_queried = (pseudo_label.size()[0] - num_selected) // args.max_iterations
    print('{} selected from {} data are initialized with pseudo label, and overall {} data has correct pseudo labels'.format(num_selected, pseudo_label.size()[0], torch.sum(mask)))
    # Begin iterations
    best_accuracy = 0
    for k in range(args.max_iterations):
        # Retrain the source models
        netF_list, netB_list, netD = retrain_model(netF_list, netB_list, netC_list, netD, optim_group, optimizer_domain, pseudo_label)
        # Evaluate performance, and update prototype
        all_logits = []
        prototype_list = []
        for i in range(len(args.src)):
            acc, logits = cal_acc_model(dset_loaders['test'], netF_list[i], netB_list[i], netC_list[i], all_targets)
            print('Accuracy on domain-{} model is {}.'.format( i+1, acc))
            all_logits.append(logits)
            # find prototype weights
            prototype_w = prototype(dset_loaders['test'], netF_list[i], netB_list[i], args.lambda_bandwidth, pseudo_label)
            prototype_list.append(prototype_w)
        # Update the pseudo label
        num_selected = pseudo_label[pseudo_label >= 0].size()[0]
        domain_weights, predictions_weighted, netW = domain_aggregation(all_logits, netW, optimizer_w, args.w_epoch)
        pseudo_label, pseudo_label_raw = update_pseudo_label(domain_weights, prototype_list, predictions_weighted, all_targets, num_selected, num_queried)
        # Ensemble accuracy
        acc_ensemble = ensemble(all_logits, all_targets,  netW)
        if best_accuracy< acc_ensemble:
            best_accuracy = acc_ensemble

        mask = pseudo_label_raw == all_targets
        num_selected = pseudo_label[pseudo_label >= 0].size()[0]
        print('{} selected from {} data are initialized with pseudo label, and overall {} data has correct pseudo labels'.format(num_selected, pseudo_label.size()[0], torch.sum(mask)))
        torch.cuda.empty_cache()

        log_str = '\nIteration: {}, Best-Accuracy: {:.2f}%'.format(k, best_accuracy*100)
        args.out_file.write(log_str)
        args.out_file.flush()
        print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

###################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSFDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--max_epoch', type=int, default=3, help="max iterations of retraining model")
    parser.add_argument('--w_epoch', type=int, default=20000, help="max iterations of optimizing domain weights")
    parser.add_argument('--max_iterations', type=int, default=20, help="max iterations of algorithm")
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--upper_bound', type=str, default='False')
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=1, help="number of workers")
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1 * 1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    # Hyper-parameters
    parser.add_argument('--lambda_supervised', type=float, default=0.1)
    parser.add_argument('--lambda_confidence', type=float, default=0.5)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_MI', type=float, default=1.0)
    parser.add_argument('--lambda_bandwidth', type=float, default=10)
    parser.add_argument('--prototype_itr', type=float, default=2)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='ckps/adapt_ours')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = './data/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i][0].upper()))
    print(args.output_dir_src)

    args.output_dir = osp.join(args.output, args.dset, names[args.t][0].upper())
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    file_name =  '_'.join([f'itr{args.max_iterations}', f'supervised{args.lambda_supervised}', f'threshold{args.lambda_confidence}', 'log.txt'])
    args.out_file = open(osp.join(args.output_dir, file_name), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    train_target(args)
