import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

import sys
sys.path.append("..")
from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)
model_name = 'Retain'
resume_path = 'Epoch_50_JA_0.4952_DDI_0.08157.model'

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--target_ddi', type=float, default=0.06, help='target ddi')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--cuda', type=int, default=0, help='which cuda')
parser.add_argument('--round', type=int, default=1, help="train round number")

args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        if len(input) < 2: continue
        for i in range(1, len(input)):
            target_output = model(input[:i])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prob
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.4] = 1
            y_pred_tmp[y_pred_tmp < 0.4] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 =\
                multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        
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

        model = Retain(voc_size, device=device)
        # model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, args.resume_path), 'rb')))

        print('parameters', get_n_params(model))
        optimizer = Adam(model.parameters(), args.lr)
        model.to(device=device)

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
                if len(input) < 2: continue

                loss = 0
                for i in range(1, len(input)):
                    target = np.zeros((1, voc_size[2]))
                    target[:, input[i][2]] = 1

                    output_logits = model(input[:i])
                    loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).to(device))

                optimizer.zero_grad()
                loss.backward()
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
            torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, str(current_round), \
                'Epoch_{}_{}.model'.format(epoch, ja)), 'wb'))

            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

            print ('best_epoch: {}'.format(best_epoch))

        history_train['time'].append(np.mean(metrics))
        dill.dump(history_train, open(os.path.join('saved', args.model_name, 'history_train.pkl'), 'wb'))

        # test module
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

if __name__ == '__main__':
    main()
