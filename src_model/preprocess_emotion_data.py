# preprocess the emotion datasets (goemotions, hurricane)

import os
import random
import json
import pickle
from collections import Counter
from configs import PATH_HURRICANE, PATH_GOEMOTIONS, PATH_MAPPINGS, PATH_PROCESSED

RATIO_TRAIN, RATIO_TEST, RATIO_DEV = 0.7, 0.2, 0.1
EMO_MAPPING = "ekman"
random.seed(11)

def load_tsv(path):
    with open(path,"r") as file:
        lines = [l.strip() for l in file.readlines()]
    lines = [l.split("\t") for l in lines if l != ""]
    return lines

def load_lines(path):
    with open(path, "r") as file:
        lines = [l.strip() for l in file.readlines() if l.strip()!=""]
    return lines

def load_mapping(path):
    with open(path, "r") as file:
        return json.loads(file.read())

def load_emotion_idx(path, reverse=False):
    emotions = load_lines(path)
    if reverse:
        emotion_idx = {i: e for i, e in enumerate(emotions)}
    else:
        emotion_idx = {e: i for i, e in enumerate(emotions)}
    return emotion_idx


def _import_go(path: str, emo_idx, emo_mapping, new_emo_idx):
    data = load_tsv(path)
    new_data = []
    for text, idxs, _ in data:
        idxs = idxs.split(",")
        emotions = set([emo_mapping[emo_idx[int(i)]]
                        for i in idxs if i != "27"])
        if len(emotions) > 0:
            idxs = [new_emo_idx[e] for e in emotions]
            # idxs = sorted(idxs)
            idxs = convert_idxs_to_one_hot(idxs, len(new_emo_idx))
            new_data.append((text, idxs))
    return new_data


def import_go(path=PATH_GOEMOTIONS, mapping="ekman", path_mappings=PATH_MAPPINGS):
    data = {}
    emotion_idx = load_emotion_idx(
        os.path.join(path, "emotions.txt"), reverse=True)
    emotion_mapping = load_mapping(os.path.join(
        path_mappings, "go-"+mapping+".json"))
    new_emotion_idx = load_emotion_idx(
        os.path.join(path_mappings, mapping+".txt"))

    if os.path.isfile(os.path.join(path, "train.tsv")):
        for split in ["train", "test", "dev"]:
            data[split] = _import_go(os.path.join(
                path, split+".tsv"), emotion_idx, emotion_mapping, new_emotion_idx)
    else:
        raise NotImplementedError
    return data


def count_annotations(line, mapping):
    cnt = Counter()
    for annot_idx, annot in enumerate(line["annotations"].values()):
        mapped_annot = [mapping[k] for k, v in annot.items() if v]
        mapped_annot = list(set(mapped_annot))
        cnt.update(mapped_annot)
        # print(f"Annotator {annot_idx}: "+str([f"{mapping[k]} ({k})" for k, v in annot.items() if v]))
    return cnt


def convert_idxs_to_one_hot(idxs, vec_size):
    one_hot = [0]*vec_size
    for i in idxs:
        one_hot[i] = 1
    return one_hot


def process_raw_hurricane(lines, emo_mapping, new_emo_idx, annot_threshold=3):
    new_lines = []
    for tweet_dict in lines:
        text = tweet_dict['text']
        idxs = []
        counts = count_annotations(tweet_dict, emo_mapping)
        for emo, emo_counts in counts.items():
            if emo_counts >= annot_threshold:
                idxs.append(new_emo_idx[emo])
        # if len(idxs) > 0:
        if len(idxs) >= 0:
            idxs = convert_idxs_to_one_hot(idxs, len(new_emo_idx))
            # idxs = sorted(idxs)
            new_lines.append((text, idxs))
    return new_lines


def import_one_hurricane_file(path):
    lines = load_lines(path)
    lines = [json.loads(l) for l in lines]
    return lines


def import_hurricane(path=PATH_HURRICANE, mapping="ekman", path_mappings=PATH_MAPPINGS):
    emotion_mapping = load_mapping(os.path.join(
        path_mappings, "hurricane-"+mapping+".json"))
    new_emotion_idx = load_emotion_idx(
        os.path.join(path_mappings, mapping+".txt"))

    lines = []
    for hurricane_file in os.listdir(path):
        lines.extend(import_one_hurricane_file(
            os.path.join(PATH_HURRICANE, hurricane_file)))

    lines = process_raw_hurricane(lines, emotion_mapping, new_emotion_idx)
    random.shuffle(lines)
    num_train, num_test = int(RATIO_TRAIN*len(lines)
                              ), int(RATIO_TEST*len(lines))
    data = {}
    data["train"], data["dev"], data["test"] = lines[:num_train], lines[num_train:-
                                                                        num_test], lines[-num_test:]
    return data


def import_one_hurricane_file(path):
    lines = load_lines(path)
    lines = [json.loads(l) for l in lines]
    return lines

def filter_hurricane(data):
    new_data = {}
    for split, split_data in data.items():
        split_data = [d for d in split_data if 1 in d[1]]
        new_data[split] = split_data
    return new_data


datasets = {}
datasets["go"] = import_go(mapping=EMO_MAPPING)
datasets["hurricane"] = import_hurricane(mapping=EMO_MAPPING)


if not os.path.isdir(PATH_PROCESSED):
    os.makedirs(PATH_PROCESSED)
    
with open(os.path.join(PATH_PROCESSED, f"hurricane_unfiltered-{EMO_MAPPING}.pkl"), "wb") as file:
    pickle.dump(datasets['hurricane'], file)

for name, data in datasets.items():
    print(
        f"{name}\t train:{len(data['train'])} test:{len(data['test'])} dev:{len(data['dev'])}")
    if name == "hurricane":
        data = filter_hurricane(data)
        datasets[name] = data
        print("after filtering out samples without any label:")
        print(
            f"{name}\t train:{len(data['train'])} test:{len(data['test'])} dev:{len(data['dev'])}")
    with open(os.path.join(PATH_PROCESSED, f"{name}-{EMO_MAPPING}.pkl"), "wb") as file:
        pickle.dump(data, file)

# print(random.sample(datasets["go"]["test"], 5))
# print(random.sample(datasets["hurricane"]["test"], 5))
