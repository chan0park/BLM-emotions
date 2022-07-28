import pandas as pd
import numpy as np
import torch
import pickle
import random
import os
# import time
import math
import json

from torch.nn import BCELoss, NLLLoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

from args import args

from main_functions import import_data, import_fewshot_data, make_data, data_to_dataloader, copy_file, delete_models, load_best_fewshot_model, load_best_model, load_best_pre_model, load_optimizer, save_model, load_model, load_optimizer_dom

from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
from configs import BERT_DIR, EMO_EKMAN

import logging

use_cuda = torch.cuda.is_available() and (not args.no_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
args.n_gpu = torch.cuda.device_count()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

emotions = EMO_EKMAN
assert args.target_emotion in emotions, f"target emotion {args.target_emotion} is not in the list of {args.emotion} ({emotions})"

num_labels = 2
args.n_cls = 2
# emo_mapping_i2e = {i: e for i, e in enumerate(emotions)}


def freeze_bert_low_layers(layers=range(0, 8), freeze_emb=True):
    for name, param in model.named_parameters():
        if any([f".{num}." in name for num in layers]):
            param.requires_grad = False
        elif freeze_emb and "embeddings" in name:
            param.requires_grad = False


def freeze_or_unfreeze_bert_layers(layers=range(0, 12), freeze=True):
    for name, param in model.named_parameters():
        if "bert." in name:
            # if any([f".{num}." in name for num in layers]):
            param.requires_grad = False if freeze else True


def freeze_or_unfreeze_domain(freeze=False):
    for name, param in model.named_parameters():
        if "classifier_domain" in name:
            param.requires_grad = False if freeze else True


def freeze_or_unfreeze_cls(freeze=False):
    for name, param in model.named_parameters():
        if "classifier_cls" in name:
            param.requires_grad = False if freeze else True


def pre_train(model, src_train, src_dev, optimizer):
    if args.verbose:
        logging.info("Pre-training started")

    freeze_or_unfreeze_bert_layers(freeze=False)
    freeze_or_unfreeze_cls(freeze=False)
    freeze_or_unfreeze_domain(freeze=True)
    # optimizer = load_optimizer(args.lr)

    num_step, num_ep, tr_loss, num_tr_examples = 0, 1, 0, 0
    dev_decreased, best_dev_f1 = 0.0, 0.0
    PATIENCE = 3
    F1_THRE = 10
    train_loss_set = []

    len_train = len(src_train)
    src_train_iter = iter(src_train)

    model.train()
    while dev_decreased <= PATIENCE:
        # for step, src_batch in enumerate(src_train):
        try:
            src_batch = next(src_train_iter)
        except StopIteration:
            src_train_iter = iter(src_train)
            src_batch = next(src_train_iter)
            num_ep += 1

        num_step += 1
        batch = tuple(t.to(device) for t in src_batch)
        input_ids, input_mask, labels, token_types = batch
        optimizer.zero_grad()

        # Forward pass for multilabel classification
        logits_class, _ = model(
            input_ids, token_type_ids=None, attention_mask=input_mask)
        src_loss_class = loss_fn_class(logits_class,labels)

        train_loss_set.append(src_loss_class.item())
        src_loss_class.backward()
        # optimizer.step()
        optimizer_step(optimizer)
        # scheduler.step()

        tr_loss += src_loss_class.item()
        num_tr_examples += input_ids.size(0)
        if num_step % args.pre_every == 0 and args.verbose:
            logging.info(
                f"Train[p]\t{num_ep: >2}-{num_step: <6}\tSRC-CLS: {tr_loss/num_step:.3f}")
            res, _ = validate(src_dev, 0.5, bool_domain=False)
            dev_f1 = res[0]
            if dev_f1 >= best_dev_f1:
                dev_decreased = 0
                best_dev_f1 = dev_f1
                if dev_f1 > F1_THRE and args.save:
                    delete_models("pretrained_", args.save_path)
                    logging.info(f"saving trained models to {args.save_path}/pretrained_{dev_f1:.2f}.pt\n")
                    save_model(args, f"pretrained_{dev_f1:.2f}.pt", model)
            elif num_step > 2500:
                logging.info("dev f1 decreased.")
                dev_decreased += 1
            model.train()

def optimizer_step(opt, bool_clip=True):
    if bool_clip:
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,
                                            model.parameters()), args.clip)
    opt.step()


def get_curriculum_coeff(cur_step, init=0, max_coeff=1, min_step=0, max_step=10000):
    cur_tot_step = (args.cur_epoch-1) * args.step_joint + cur_step
    max_step = args.epoch * args.step_joint
    if cur_tot_step < min_step:
        return 0
    else:
        return min(max_coeff, max_coeff*math.sqrt(cur_tot_step/max_step*(1-init**2))+init)

def get_curriculum_coeff_epoch(init=0, max_coeff=2, max_epoch=2):
    cur_epoch = args.cur_epoch-1
    return min(max_coeff, max_coeff*math.sqrt(cur_epoch/max_epoch*(1-init**2))+init)

def train_curriculum(model, dataloaders, optimizer, loss_ratio):
    if args.curriculum:
        print(f"training a curriculum model")
    else:
        print(f"training a model")
    src_train, src_dev, tgt_train, tgt_dev = dataloaders

    freeze_or_unfreeze_bert_layers(freeze=False)
    freeze_or_unfreeze_cls(freeze=False)
    freeze_or_unfreeze_domain(freeze=False)
    optimizer_dom = load_optimizer_dom(model, args.lr)

    num_step, num_tr_examples = 0, 0
    tr_src_loss_class, tr_src_loss_domain, tr_tgt_loss_domain,  = 0, 0, 0
    running_tr_src_loss_class = 0
    num_src_ep, num_tgt_ep = 1, 1
    train_loss_set = []

    src_train_iter = iter(src_train)
    tgt_train_iter = iter(tgt_train)

    model.train()
    for step in range(args.step_joint):
        try:
            src_batch = next(src_train_iter)
        except StopIteration:
            if args.initialize_dom:
                if args.n_gpu > 1:
                    model.module.initialize_domain()
                else:
                    model.initialize_domain()
            src_train_iter = iter(src_train)
            src_batch = next(src_train_iter)
            num_src_ep += 1
        try:
            tgt_batch = next(tgt_train_iter)
        except StopIteration:
            tgt_train_iter = iter(tgt_train)
            tgt_batch = next(tgt_train_iter)
            num_tgt_ep += 1

        num_step += 1
        optimizer.zero_grad()
        optimizer_dom.zero_grad()

        src_batch = tuple(t.to(device) for t in src_batch)
        if len(src_batch) == 4:
            input_ids, input_mask, labels, token_types = src_batch
            probs = None
        elif len(src_batch) == 5:
            input_ids, input_mask, labels, token_types, probs = src_batch
        domain_labels = torch.Tensor([[args.max_dom_prob, 1.0-args.max_dom_prob]]*labels.size(0)).to(device)
        if args.curriculum:
            # Forward pass for multilabel classification
            _, logits_domain = model(
                input_ids, token_type_ids=None, attention_mask=input_mask, bool_domain=True, bool_class=False)
            pred_domain = softmax(logits_domain, dim=-1)
            src_loss_domain = loss_fn_domain_joint(pred_domain, domain_labels)

            tgt_batch = tuple(t.to(device) for t in tgt_batch)
            tgt_input_ids, tgt_input_mask, tgt_labels, token_types = tgt_batch
            domain_labels = torch.Tensor([[1.0-args.max_dom_prob, args.max_dom_prob]]*tgt_labels.size(0)).to(device)
            # Forward pass for multilabel classification
            _, logits_domain = model(
                tgt_input_ids, token_type_ids=None, attention_mask=tgt_input_mask, bool_domain=True, bool_class=False)
            pred_domain = softmax(logits_domain, dim=-1)
            tgt_loss_domain = loss_fn_domain_joint(pred_domain, domain_labels)
            loss_dom = loss_ratio * (src_loss_domain + tgt_loss_domain)
            loss_dom.backward()
            optimizer_step(optimizer_dom, bool_clip=False)

        logits_class, logits_domain = model(
            input_ids, token_type_ids=None, attention_mask=input_mask, bool_domain=True)
        pred_domain = softmax(logits_domain, dim=-1)
        logits_class= logits_class.view(-1,num_labels)
        src_loss_class = loss_fn_class_noreduce(logits_class,labels)

        if args.curriculum:
            probs = pred_domain[:,1]
            if args.fix_coeff:
                lambda_curriculum = 1
            else:
                lambda_curriculum = get_curriculum_coeff(step, min_step=args.min_step, max_coeff=args.max_coeff)
            src_loss_class = src_loss_class * (probs**lambda_curriculum)
        loss = src_loss_class.mean()

        # train_loss_set.append(loss.item())
        loss.backward()
        optimizer_step(optimizer)


        tr_src_loss_class += loss.item()
        running_tr_src_loss_class += loss.item()
        if args.curriculum:
            tr_src_loss_domain += src_loss_domain.item()
            tr_tgt_loss_domain += tgt_loss_domain.item()
        num_tr_examples += input_ids.size(0)
        if num_step % args.joint_every == 0 and args.verbose:
            logging.info(
                f"Train[j]\t{num_step: <6}\tSRC-CLS: {tr_src_loss_class/num_step:.3f}\tSRC-DOM: {tr_src_loss_domain/num_step:.3f}\tTGT-DOM: {tr_tgt_loss_domain/num_step:.3f}")
            res_src, _ = validate(
                src_dev, 0.5, bool_domain=True, domain_label=0)
            res_tgt, _ = validate(
                tgt_dev, 0.5, bool_domain=True, domain_label=1)
            res_src_f1 = res_src[0]
            res_tgt_f1 = res_tgt[0]
            res_src_loss = res_src[-1]
            res_tgt_loss = res_tgt[-1]
            
            if args.save:
                writer.add_scalar('[j]loss/src-train', running_tr_src_loss_class/args.joint_every, (args.step_joint * (args.cur_epoch-1))+num_step)
                writer.add_scalar('[j]loss/src-dev', res_tgt_loss, (args.step_joint * (args.cur_epoch-1))+num_step)
                writer.add_scalar('[j]loss/tgt-dev', res_tgt_loss, (args.step_joint * (args.cur_epoch-1))+num_step)
                writer.add_scalar('[j]f1/src-dev', res_src_f1, (args.step_joint * (args.cur_epoch-1))+num_step)
                writer.add_scalar('[j]f1/tgt-dev', res_tgt_f1, (args.step_joint * (args.cur_epoch-1))+num_step)
            running_tr_src_loss_class = 0

            if args.use_src_f1:
                if res_src_f1 > args.best_zero_src_f1:
                    args.best_zero_src_f1 = res_src_f1
                    args.best_zero_tgt_f1 = res_tgt_f1
            else:
                if res_tgt_f1 > args.best_zero_tgt_f1:
                    args.best_zero_src_f1 = res_src_f1
                    args.best_zero_tgt_f1 = res_tgt_f1

            res_f1 = res_tgt_f1 if args.few_shot else res_src_f1
            if res_f1 >= args.best_f1:
                args.best_f1 = res_f1
                args.best_src_f1 = res_src_f1
                args.best_tgt_f1 = res_tgt_f1
                args.best_tgt_epoch = args.cur_epoch
                logging.info(
                    f"Best F1 Found -- src:{res_src_f1:.3f}, tgt: {res_tgt_f1:.3f}")
                if args.save:
                    if args.do_adv and args.src_add_ratio>0:
                        copy_file(args.save_path, "src_add_temp","src_add_samples")
                    delete_models("best_", args.save_path)
                    logging.info(f"saving trained models to {args.save_path}/best_{res_tgt_f1:.2f}_{res_src_f1:.2f}.pt\n")
                    save_model(args, f"best_{res_tgt_f1:.2f}_{res_src_f1:.2f}.pt", model)

            model.train()
    return model


def filter_data_by_thre(src_train, thre):
    num_original = len(src_train['labels'])
    dataloader = data_to_dataloader(
        src_train, batch_size=args.batch_size, bool_valid=True)
    new_data, new_src_train = {}, []
    num_filtered = 0
    filtered = []
    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, input_mask, labels, token_types = batch
        input_id_tensor = input_ids.to(device)
        att_mask_tensor = input_mask.to(device)
        with torch.no_grad():
            _, logits_domain = model(
                input_id_tensor, token_type_ids=None, attention_mask=att_mask_tensor, bool_domain=True)
            pred_domain = softmax(logits_domain, dim=-1).detach().cpu()
            keep_index = [bool(p[0] <= thre) for p in pred_domain]
            filter_index = [bool(p[0] > thre) for p in pred_domain]
        num_filtered += sum(filter_index)
        filtered_input = input_ids[filter_index].tolist()
        filtered_label = labels[filter_index].tolist()
        filtered.extend(list(zip(filtered_input, filtered_label)))

        input_ids = input_ids[keep_index].tolist()
        input_mask = input_mask[keep_index].tolist()
        labels = labels[keep_index].tolist()
        token_types = token_types[keep_index].tolist()
        new_src_train.extend(
            list(zip(input_ids, input_mask, labels, token_types)))
    if num_filtered == 0:
        logging.info(
            f"num src training filtered: {num_filtered}/{num_original} ({num_filtered/num_original*100:.1f}%)")
        return src_train
    input_ids, att_masks, labels, type_ids = zip(*new_src_train)
    new_data = make_data(input_ids, type_ids, att_masks, labels)
    write_filtered(filtered, num_original)
    return new_data

def write_src_add(src_add):
    num_add = len(src_add['labels'])
    torch.save(src_add, f"{args.save_path}/src_add_temp.pt")
    with open(f"{args.save_path}/src_add_temp.txt", "w") as file:
        file.write("\n".join([" ".join(tokenizer.convert_ids_to_tokens(
            ws, skip_special_tokens=True))+"\t"+vec2text_label(label) for ws, label in zip(src_add['input_ids'].tolist(), src_add['labels'].tolist())]))

    if args.verbose:
        logging.info(
            f"src additional {num_add} training saved to {args.save_path}/src_add_temp.txt")
        # rand_examples = random.sample(src_add, min(2, num_add))
        # rand_examples = "\n".join([" ".join(tokenizer.convert_ids_to_tokens(
        #     ws, skip_special_tokens=True))+"\t"+vec2text_label(label) for ws, label in rand_examples])
        # logging.info("src addition examples:")
        # logging.info(rand_examples)

def write_filtered(filtered, num_original):
    num_filtered = len(filtered)
    with open(f"{args.save_path}/filtered_examples.txt", "a") as file:
        file.write("\n".join([" ".join(tokenizer.convert_ids_to_tokens(
            ws, skip_special_tokens=True))+"\t"+vec2text_label(label) for ws, label in filtered])+f"\nEpoch{args.cur_epoch}-----------------------\n")
    if args.verbose:
        logging.info(
            f"num src training filtered: {num_filtered}/{num_original} ({num_filtered/num_original*100:.1f}%)")
        logging.info("filtered examples:")
        rand_examples = random.sample(filtered, min(2, num_filtered))
        rand_examples = "\n".join([" ".join(tokenizer.convert_ids_to_tokens(
            ws, skip_special_tokens=True))+"\t"+vec2text_label(label) for ws, label in rand_examples])
        logging.info(rand_examples)

def write_examples_prob(data):
    data = [(ws, l, p) for ws, _, l, _, p in data]
    file_path = f"{args.save_path}/examples_prob_epoch{args.cur_epoch}.txt"
    with open(file_path, "w") as file:
        file.write("\n".join([" ".join(tokenizer.convert_ids_to_tokens(
            ws, skip_special_tokens=True))+"\t"+vec2text_label(label)+"\t"+str(float(prob)) for ws, label, prob in data]))
    if args.verbose:
        logging.info(
            f"sorted training examples are saved to {file_path}")

def filter_data_by_ratio(src_train, ratio):
    num_original = len(src_train['labels'])
    dataloader = data_to_dataloader(
        src_train, batch_size=args.batch_size, bool_valid=True)
    new_data, new_src_train = {}, []
    pred_domains,pred_domains_tgt = [], []
    num_filtered = 0
    filtered = []
    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        if len(batch) == 4:
            input_ids, input_mask, labels, token_types = batch
        elif len(batch) == 5:
            input_ids, input_mask, labels, token_types, _ = batch
        input_id_tensor = input_ids.to(device)
        att_mask_tensor = input_mask.to(device)
        with torch.no_grad():
            _, logits_domain = model(
                input_id_tensor, token_type_ids=None, attention_mask=att_mask_tensor, bool_domain=True, bool_class=False)
            pred_domain = softmax(logits_domain, dim=-1).detach().cpu()

        pred_domain_src = [p[0] for p in pred_domain]
        pred_domain_tgt = [p[1] for p in pred_domain]
        input_ids = input_ids.tolist()
        input_mask = input_mask.tolist()
        labels = labels.tolist()
        token_types = token_types.tolist()

        pred_domains.extend(pred_domain_src)
        pred_domains_tgt.extend(pred_domain_tgt)
        new_src_train.extend(
            list(zip(input_ids, input_mask, labels, token_types)))
    
    pred_domain_args = np.argsort(pred_domains)
    num_filtered = int(len(pred_domains) * ratio)
    
    if num_filtered > 0:
        filter_index = pred_domain_args[-num_filtered:][::-1]
        filtered = [(new_src_train[i][0], new_src_train[i][2]) for i in filter_index]
        write_filtered(filtered, num_original)
        keep_index = pred_domain_args[:-num_filtered]
    else:
        keep_index = pred_domain_args

    new_src_train = [new_src_train[i]+(pred_domains_tgt[i],) for i in keep_index]
    write_examples_prob(new_src_train)

    input_ids, att_masks, labels, type_ids, probs = zip(*new_src_train)
    new_data = make_data(input_ids, type_ids, att_masks, labels, probs)
    return new_data


def initialize_adv(model):
    if args.n_gpu > 1:
        model.module.initialize_domain()
    else:
        model.initialize_domain()
    return model

def train_src(model, src_train, src_dev, tgt_train, tgt_dev):
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory
    src_train_dataloader = data_to_dataloader(
        src_train, batch_size=args.batch_size)
    src_dev_dataloader = data_to_dataloader(
        src_dev, batch_size=args.batch_size, bool_valid=True)
    tgt_train_dataloader = data_to_dataloader(
        tgt_train, batch_size=args.batch_size)
    tgt_dev_dataloader = data_to_dataloader(
        tgt_dev, batch_size=args.batch_size, bool_valid=True)

    freeze_or_unfreeze_bert_layers(freeze=False)
    freeze_or_unfreeze_cls(freeze=False)
    freeze_or_unfreeze_domain(freeze=True)
    optimizer_cls, optimizer_dom = load_optimizer(model, args.lr)

    if args.do_adv and args.skip_adv_initialize:
        freeze_or_unfreeze_bert_layers(freeze=True)
        freeze_or_unfreeze_cls(freeze=True)
        freeze_or_unfreeze_domain(freeze=False)
        optimizer_adv = load_optimizer(model, args.lr)
    else:
        optimizer_adv = None
        

    # pre-train until converge
    if not args.skip_pre:
        pre_train(model, src_train_dataloader, src_dev_dataloader, optimizer_cls)
        model = load_best_pre_model(model, args.save_path)

    dataloaders = (src_train_dataloader, src_dev_dataloader,
                   tgt_train_dataloader, tgt_dev_dataloader)
    best_tgt_dev_f1, tgt_dev_f1_decreased, PATIENCE = 0.0, 0, 3

    if not args.skip_alt:
        for i in range(args.cur_epoch, args.cur_epoch+args.epoch):
            args.cur_epoch = i
            logging.info(f"\nEpoch {i}")

            model = train_curriculum(model, dataloaders, optimizer_cls, args.loss_ratio)

            if args.initialize_dom:
                model = initialize_adv(model)
        if args.verbose:
            print("--------------------")
            logging.info(
                f"best zero shot performance -- src:{args.best_zero_src_f1}, tgt:{args.best_zero_tgt_f1}")
            logging.info(
                f"best model selected -- src:{args.best_src_f1}, tgt:{args.best_tgt_f1}")
            print("--------------------")


def validate(dataloader, threshold=0.5, bool_domain=True, domain_label=None, test=False):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Variables to gather full output
    val_loss, num_step =  0, 0
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []
    pred_doms = []

    # Predict
    for i, batch in enumerate(dataloader):
        num_step += 1
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask, labels, b_token_types = batch
        with torch.no_grad():
            logit_pred, logit_dom = model(input_ids, token_type_ids=None,
                                          attention_mask=input_mask, bool_domain=bool_domain)
            loss_class = loss_fn_class(logit_pred.view(-1, num_labels),labels)
            val_loss += loss_class.item()

            pred_label = torch.sigmoid(logit_pred)
            logit_pred = logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
            input_ids = input_ids.to('cpu')

            if bool_domain:
                pred_dom = softmax(logit_dom, dim=-1)
                pred_dom = pred_dom.to('cpu').tolist()
                pred_doms.extend(pred_dom)
                tokenized_texts.extend(
                    [" ".join(tokenizer.convert_ids_to_tokens(ws, skip_special_tokens=True)) for ws in input_ids])
            logit_preds.append(logit_pred)
            true_labels.append(labels)
            pred_labels.append(pred_label)

    val_loss = val_loss/num_step
    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate Accuracy
    # threshold = 0.50
    if threshold is None:
        best_thre, best_f1, best_acc = 0.0, 0.0, 0.0
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_bools = [pl > threshold for pl in pred_labels]
            true_bools = [tl == 1 for tl in true_labels]
            val_f1_accuracy = f1_score(
                true_bools, pred_bools)*100
            val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100
            if val_f1_accuracy > best_f1:
                best_f1 = val_f1_accuracy
                best_acc = val_flat_accuracy
                best_thre = threshold
    else:
        best_thre = threshold
        pred_bools = [pl[1] > pl[0] for pl in pred_labels]
        true_bools = [tl == 1 for tl in true_labels]
        best_f1 = f1_score(true_bools, pred_bools)*100
        best_acc = accuracy_score(true_bools, pred_bools)*100

        if bool_domain:
            pred_dom_bools = [(pd_1 > pd_0) for pd_0, pd_1 in pred_doms]
            true_dom_bools = [(domain_label == 1) for pd in pred_doms]
            best_dom_f1 = f1_score(
                true_dom_bools, pred_dom_bools)*100
            best_dom_acc = accuracy_score(true_dom_bools, pred_dom_bools)*100
        else:
            best_dom_f1, best_dom_acc = 0, 0

    if bool_domain:
        logging.info(
            f"\tValid\tF1-D: {best_dom_f1:.3f}\tAcc-D: {best_dom_acc:.3f}\tF1: {best_f1:.3f}\tAcc: {best_acc:.3f}")
        rand_text, rand_pred = random.choice(
            list(zip(tokenized_texts, pred_doms)))
        logging.info(
            f"\t{rand_text}\t[{rand_pred[0]:.2f},{rand_pred[1]:.2f}]\t{int(rand_pred[1] > rand_pred[0])}")
    else:
        if test:
            logging.info(f"\tTest\tF1: {best_f1:.3f}\tAcc: {best_acc:.3f}")
        else:
            logging.info(f"\tValid\tF1: {best_f1:.3f}\tAcc: {best_acc:.3f}")
    return (best_f1, best_acc, best_dom_f1, best_dom_acc, val_loss), best_thre
    # return (best_f1, best_acc, best_dom_f1, best_dom_acc, tokenized_texts, pred_labels, true_labels), best_thre

def vec2text_label(labels):
    if labels == 1:
        return args.target_emotion
    else:
        return "neutral"
    # text_labels = []
    # for i, lbl in enumerate(labels):
    #     if lbl == 1:
    #         text_labels.append(emo_mapping_i2e[i])
    # if len(text_labels) == 0:
    #     return "neutral"
    # return ",".join(text_labels)

def write_batch(batch_input_ids, pred_labels, gold_labels, pred_probs):
    sents = [tokenizer.convert_ids_to_tokens(
        sent, skip_special_tokens=True) for sent in batch_input_ids]
    sents = [" ".join(sent) for sent in sents]
    gold_labels = [vec2text_label(label) for label in gold_labels]
    pred_labels = [vec2text_label(label) for label in pred_labels]
    pred_probs = [",".join([str(round(d, 2)) for d in p]) for p in pred_probs]

    with open(args.save_path+f"/{args.target}-test.output", "a") as file:
        for sent, gl, pl, pp in zip(sents, gold_labels, pred_labels, pred_probs):
            file.write(f"{sent}\t{gl}\t{pl}\t{pp}\n")


def eval(test_data, threshold, suffix="", save_report=True, save_output=True):
    test_data = TensorDataset(
        test_data['input_ids'], test_data['attention_masks'], test_data['labels'], test_data['token_type_ids'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.batch_size)
    model.eval()
    # logit_preds, true_labels, pred_labels, tokenized_texts, pred_bools, true_bools = [], [], [], [], [], []
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []
    for i, batch in enumerate(test_dataloader):
        batch_input_ids = batch[0].numpy()
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            # Forward pass
            b_logit_pred, _ = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
            # b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        if save_output:
        # pred_bools = [pl[1] > pl[0] for pl in pred_labels]
        # true_bools = [tl == 1 for tl in true_labels]
            pred_bool = [pl[1]>pl[0] for pl in pred_label]
            true_bool = [tl==1 for tl in b_labels]
            write_batch(batch_input_ids, pred_bool, true_bool, pred_label)

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)
    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl == 1 for tl in true_labels]
    # boolean output after thresholding
    # best_thre = threshold if threshold else args.thre
    pred_bools = [pl[1] > pl[0] for pl in pred_labels]

    test_f1_accuracy = f1_score(true_bools, pred_bools)*100
    test_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

    logging.info(
        f"Test\tF1: {test_f1_accuracy:.3f}\tAcc: {test_flat_accuracy:.3f}")

    clf_report = classification_report(
        true_bools, pred_bools, target_names=["non-"+args.target_emotion,args.target_emotion])
    logging.info(clf_report)
    if save_report:
        with open(f'{args.save_path}/classification_report{suffix}.txt', "w") as file:
            file.write(clf_report)
    return test_f1_accuracy, test_flat_accuracy


def train_fewshot(model, tgt_train, tgt_dev, tgt_test, src_similar_data=None, lr=1e-5):
    if src_similar_data is not None:
        logging.info(f"combining additional {len(src_similar_data['labels'])} data from source")
        tgt_train = combine_data(tgt_train, src_similar_data, tokenizer.pad_token_id)
    tgt_train_dataloader = data_to_dataloader(
        tgt_train, batch_size=args.fewshot_batch_size)
    tgt_dev_dataloader = data_to_dataloader(
        tgt_dev, batch_size=args.batch_size, bool_valid=True)
    tgt_test_dataloader = data_to_dataloader(
        tgt_test, batch_size=args.batch_size, bool_valid=True)
    # train until converge
    _train_fewshot(model, tgt_train_dataloader,
                   tgt_dev_dataloader, tgt_test_dataloader, lr)


def _train_fewshot(model, tgt_train, tgt_dev, tgt_test, lr):
    if args.verbose:
        logging.info("Few shot training started")

    freeze_or_unfreeze_bert_layers(freeze=False)
    freeze_or_unfreeze_cls(freeze=False)
    freeze_or_unfreeze_domain(freeze=True)
    optimizer, _ = load_optimizer(model, lr)

    num_step, num_ep, tr_loss, running_tr_loss, num_tr_examples = 0, 1, 0, 0, 0
    dev_decreased, best_dev_f1, last_f1 = 0.0, 0.0, 0.0
    PATIENCE = 3
    train_loss_set = []

    len_train = len(tgt_train)
    tgt_train_iter = iter(tgt_train)

    model.train()
    while dev_decreased < PATIENCE and num_step <= 2000:
        # for step, tgt_batch in enumerate(src_train):
        try:
            tgt_batch = next(tgt_train_iter)
        except StopIteration:
            tgt_train_iter = iter(tgt_train)
            tgt_batch = next(tgt_train_iter)
            num_ep += 1

        num_step += 1
        batch = tuple(t.to(device) for t in tgt_batch)
        input_ids, input_mask, labels, token_types = batch
        optimizer.zero_grad()

        # Forward pass for multilabel classification
        logits_class, _ = model(
            input_ids, token_type_ids=None, attention_mask=input_mask)
        src_loss_class = loss_fn_class(logits_class.view(-1, num_labels),
                                       labels)

        # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        train_loss_set.append(src_loss_class.item())
        src_loss_class.backward()
        # optimizer.step()
        optimizer_step(optimizer)
        # scheduler.step()

        tr_loss += src_loss_class.item()
        running_tr_loss += src_loss_class.item()
        num_tr_examples += input_ids.size(0)
        if num_step % args.fewshot_every == 0 and args.verbose:
            logging.info(
                f"Train[f]\t{num_ep: >2}-{num_step: <6}\tSRC-CLS: {tr_loss/num_step:.3f}")
            res, _ = validate(tgt_dev, 0.5, bool_domain=False)
            res_test, _ = validate(tgt_test, 0.5, bool_domain=False, test=True)
            dev_f1 = res[0]
            test_f1 = res_test[0]
            dev_loss = res[-1]
            if args.save:
                writer.add_scalar('[f]loss/train', running_tr_loss/args.fewshot_every, num_step)
                writer.add_scalar('[f]loss/valid', dev_loss, num_step)
                writer.add_scalar('[f]f1/valid', dev_f1, num_step)
                writer.add_scalar('[f]f1/test', test_f1, num_step)
            running_tr_loss = 0

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                # if args.save and num_step > 200:
                if args.save:
                    delete_models(f"fewshot_{lr}_", args.save_path)
                    logging.info(f"saving trained models to {args.save_path}/fewshot_{lr}_{dev_f1:.2f}_{test_f1:.2f}.pt\n")
                    save_model(args, f"fewshot_{lr}_{dev_f1:.2f}_{test_f1:.2f}.pt", model)

            if num_step > 500:
                if dev_f1 > last_f1:
                    dev_decreased = 0
                else:
                    logging.info("dev f1 decreased.")
                    dev_decreased += 1
            last_f1 = dev_f1
            model.train()


# actual code starts from here

# logging (tensorboard, log.txt)
logging_handlers = [logging.StreamHandler()]
if args.save:
    writer = SummaryWriter('runs/'+args.suffix[1:]+"_"+args.timestamp[-5:])
    logging_handlers += [logging.FileHandler(f'{args.save_path}/log.txt')]
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=logging_handlers
)

# load model & tokenizer
model, tokenizer = load_model(args, BERT_DIR, device)
if args.verbose:
    if args.bert_path != None:
        logging.info(
            f"* loaded a bert model from {args.bert_path} and its tokenizer")
        if args.load_from:
            logging.info(
                f"* loaded a bert params from {args.load_from}")
    else:
        logging.info(f"* loaded {args.bert_model} model and its tokenizer")

# load data
emo_mapping_e2i = {e: i for i, e in enumerate(emotions)}
target_emotion_idx = emo_mapping_e2i[args.target_emotion]
data, src_data, tgt_data  = import_data(args.data_path, args.emotion, target_emotion_idx, tokenizer)
src_train, src_dev, src_test = src_data
tgt_train, tgt_dev, tgt_test = tgt_data


if args.few_shot:
    few_shot_train, num_src_add = import_fewshot_data(data[args.target], args.num_few_shot, target_emotion_idx, tokenizer, src_add_ratio=args.src_add_ratio)

# define loss functions
loss_fn_class = CrossEntropyLoss()
loss_fn_class_noreduce = CrossEntropyLoss(reduction="none")
loss_fn_domain = CrossEntropyLoss()
loss_fn_domain_joint = BCELoss()

# start training
zero_src, zero_tgt, few_src, few_tgt = None, None, None, None
if not args.skip_training:
    train_src(model, src_train, src_dev, tgt_train, tgt_dev)
    model = load_best_model(model, args.save_path, args.load_from)

    zero_src = eval(src_test, args.thre, suffix="_zero_src")
    zero_tgt = eval(tgt_test, args.thre, suffix="_zero_tgt")

if args.few_shot:
    if args.do_adv and args.src_add_ratio > 0:
        src_similar_data = torch.load(args.save_path+f"/src_add_samples.pt")
    else:
        src_similar_data = None
    train_fewshot(model, few_shot_train, tgt_dev, tgt_test, src_similar_data, lr=1e-5)
    model = load_best_fewshot_model(model, args.save_path, lr=1e-5)

    few_src = eval(src_test, args.thre, suffix="_few_src")
    few_tgt = eval(tgt_test, args.thre, suffix="_few_tgt")

def save_final_summary(args, zero_src, zero_tgt, few_src, few_tgt):
    model_name = ""
    save_path_name = args.save_path.split("/")[-1]
    res_dict = {"model":model_name, "emo":args.target_emotion, "seed":args.seed, "bert":args.bert_path_model, 'loss_ratio': args.loss_ratio, 'num_fewshoot':args.num_few_shot}
    for res, zero_few, src_tgt in [(zero_src, "zero", "src"), (zero_tgt, "zero", "tgt"), (few_src, "few", "src"), (few_tgt, "few", "tgt")]:
        f1, acc = res if res is not None else (None, None)
        res_dict[f"{zero_few}_{src_tgt}_f1"] = f1
        res_dict[f"{zero_few}_{src_tgt}_acc"] = acc
    res_dict["savedir"] = args.save_path
    with open(args.save_path+"/summary.json", "w") as file:
        json.dump(res_dict, file)
    with open(args.summary_path+f"{save_path_name}.json", "w") as file:
        json.dump(res_dict, file)

if args.save:
    save_final_summary(args, zero_src, zero_tgt, few_src, few_tgt)
    writer.close()