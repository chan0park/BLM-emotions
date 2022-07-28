import torch
import os
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from args import args
from main_functions import import_data, load_model
from configs import BERT_DIR, EMO_EKMAN

import logging

use_cuda = torch.cuda.is_available() and (not args.no_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
args.n_gpu = torch.cuda.device_count()

emotions = EMO_EKMAN
assert args.target_emotion in emotions, f"target emotion {args.target_emotion} is not in the list of {args.emotion} ({emotions})"

num_labels = 2
args.n_cls = 2

def vec2text_label(labels):
    if labels == 1:
        return args.target_emotion
    else:
        return "neutral"

def write_batch(batch_input_ids, pred_labels, pred_probs):
    sents = [tokenizer.convert_ids_to_tokens(
        sent, skip_special_tokens=True) for sent in batch_input_ids]
    sents = [" ".join(sent) for sent in sents]
    pred_labels = [vec2text_label(label) for label in pred_labels]
    pred_probs = [",".join([str(round(d, 2)) for d in p]) for p in pred_probs]

    with open(args.save_output_path, "a") as file:
        for sent, pl, pp in zip(sents, pred_labels, pred_probs):
            file.write(f"{sent}\t{pl}\t{pp}\n")


def eval(test_data, threshold, suffix="", save_report=True, save_output=True):
    test_data = TensorDataset(
        test_data['input_ids'], test_data['attention_masks'], test_data['labels'], test_data['token_type_ids'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.batch_size)
    model.eval()
    
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
            pred_label = torch.sigmoid(b_logit_pred)
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        if save_output:
            pred_bool = [pl[1]>pl[0] for pl in pred_label]
            write_batch(batch_input_ids, pred_bool, pred_label)

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)


# actual code starts from here

# logging (tensorboard, log.txt)
args.save_path = os.path.join(args.model_path, args.suffix)
logging_handlers = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=logging_handlers
)


# load data
emo_mapping_e2i = {e: i for i, e in enumerate(emotions)}
target_emotion_idx = emo_mapping_e2i[args.target_emotion]

zero_src, zero_tgt, few_src, few_tgt = None, None, None, None
assert args.load_from.endswith(".pt")

model, tokenizer = load_model(args, BERT_DIR, device)
data, src_data, tgt_data, input_data = import_data(args.data_path, args.emotion, target_emotion_idx, tokenizer, args.input_file)
src_train, src_dev, src_test = src_data
tgt_train, tgt_dev, tgt_test = tgt_data

if args.verbose:
    if args.bert_path != None:
        logging.info(
            f"* loaded a bert model from {args.bert_path} and its tokenizer")
        if args.load_from:
            logging.info(
                f"* loaded a bert params from {args.load_from}")
    else:
        logging.info(f"* loaded {args.bert_model} model and its tokenizer")
eval(input_data, args.thre, suffix="_input")