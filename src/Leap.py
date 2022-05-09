import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict

import sys
sys.path.append("..")
from models import Leap
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params

torch.manual_seed(1203)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_name = 'Leap'
resume_path = 'saved/{}/Epoch_47_JA_0.4486_DDI_0.07061.model'.format(model_name)

if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--cuda', type=int, default=0, help='which cuda')
parser.add_argument('--round', type=int, default=1, help="train round number")

args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []

        for adm_index, adm in enumerate(input):
            output_logits = model(adm)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            output_logits = output_logits.detach().cpu().numpy()

            # prediction med set
            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path='../data/output/ddi_A_final.pkl')

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt

def main():

    # load data
    data_path = '../data/output/records_final.pkl'
    voc_path = '../data/output/voc_final.pkl'
    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    for rounds in range(args.round):
        np.random.seed(rounds)
        np.random.shuffle(data)
        split_point = int(len(data) * 2 / 3)
        data_train = data[:split_point]
        eval_len = int(len(data[split_point:]) / 2)
        data_test = data[split_point:split_point + eval_len]
        data_eval = data[split_point+eval_len:]
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

        END_TOKEN = voc_size[2] + 1

        model = Leap(voc_size, device=device)
        # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

        model.to(device=device)
        print('parameters', get_n_params(model))
        optimizer = Adam(model.parameters(), lr=args.lr)

        best_epoch, best_ja = 0, 0

        if not os.path.exists(os.path.join('saved', model_name, 'history_train.pkl')):
            history_train = defaultdict(list)
        else:
            history_train = dill.load(open(os.path.join('saved', model_name, 'history_train.pkl'), 'rb'))    
        fit_round = len(history_train['time'])
        current_round = fit_round + 1
        if not os.path.exists(os.path.join("saved", model_name, str(current_round))):
            os.makedirs(os.path.join("saved", model_name, str(current_round)))
        print('Current Round: {}'.format(current_round))

        EPOCH = 50
        for epoch in range(EPOCH):
            tic = time.time()
            print ('\nepoch {} --------------------------'.format(epoch))

            model.train()
            metrics = []
            for step, input in enumerate(data_train):
                for adm in input:

                    loss_target = adm[2] + [END_TOKEN]
                    output_logits = model(adm)
                    loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

            print ()
            tic2 = time.time() 
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch)
            train_time = time.time() - tic
            eval_time = time.time() - tic2
            metrics.append(train_time)
            print ('training time: {}, eval time: {}'.format(train_time, eval_time))
            '''
            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)
            history['med'].append(avg_med)

            if epoch >= 5:
                print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                    np.mean(history['ddi_rate'][-5:]),
                    np.mean(history['med'][-5:]),
                    np.mean(history['ja'][-5:]),
                    np.mean(history['avg_f1'][-5:]),
                    np.mean(history['prauc'][-5:])
                    ))
            '''
            torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, str(current_round),\
                'Epoch_{}_{}.model'.format(epoch, ja)), 'wb'))

            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

            print ('best_epoch: {}'.format(best_epoch))

        history_train['time'].append(np.mean(metrics))
        dill.dump(history_train, open(os.path.join('saved', args.model_name, 'history_train.pkl'), 'wb'))
        
        #Test module
        resume_path = os.path.join('saved', args.model_name, str(current_round), 'Epoch_{}_{}.model'.format(best_epoch, best_ja))
        model.load_state_dict(torch.load(open(resume_path, 'rb')))
        model.to(device=device)

        result = []
        for _ in range(20):
            tic = time.time()
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, test_sample, voc_size, 0)
            pretime = time.time() - tic
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med, pretime])
            print ('inference time: {}'.format(pretime))
        
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)
        ddi_rate_avg, ja_avg, avg_f1_avg, prauc_avg, avg_med_avg, pretime_avg \
            = mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]
        ddi_rate_std, ja_std, avg_f1_std, prauc_std, avg_med_std \
            = std[0], std[1], std[2], std[3], std[4]
        print('Epoch: {}, DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
            epoch, ddi_rate_avg, ja_avg, prauc_avg, avg_f1_avg, avg_med
            ))
            
        if not os.path.exists(os.path.join('saved', model_name, 'history_test.pkl')):
            history_test = defaultdict(list)
        else:
            history_test = dill.load(open(os.path.join('saved', model_name, 'history_test.pkl'), 'rb'))   

        history_test['pretime_avg'].append(pretime_avg)
        history_test['jaccard_avg'].append(ja_avg)
        history_test['jaccard_std'].append(ja_std)
        history_test['ddi_rate_avg'].append(ddi_rate_avg)
        history_test['ddi_rate_std'].append(ddi_rate_std)
        history_test['avg_f1_avg'].append(avg_f1_avg)
        history_test['avg_f1_std'].append(avg_f1_std)
        history_test['prauc_avg'].append(prauc_avg)
        history_test['prauc_std'].append(prauc_std)
        history_test['avg_med_avg'].append(avg_med_avg)
        history_test['avg_med_std'].append(avg_med_std)
        
        dill.dump(history_test, open(os.path.join('saved', model_name, 'history_test.pkl'), 'wb'))  


def fine_tune(fine_tune_name=''):

    # load data
    data_path = '../data/output/records_final.pkl'
    voc_path = '../data/output/voc_final.pkl'
    device = torch.device('cpu:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_A = dill.load(open('../data/output/ddi_A_final.pkl', 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    # data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, fine_tune_name), 'rb')))
    model.to(device)

    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=args.lr)
    ddi_rate_record = []

    EPOCH = 100
    for epoch in range(EPOCH):
        loss_record = []
        start_time = time.time()
        random_train_set = [random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))
                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(random_train_set)))

        if K_flag:
            print ()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test, voc_size, epoch)

    # test
    torch.save(model.state_dict(), open(
        os.path.join('saved', args.model_name, 'final.model'), 'wb'))


if __name__ == '__main__':
    main()
    # fine_tune(fine_tune_name='Epoch_1_JA_0.2765_DDI_0.1158.model')
