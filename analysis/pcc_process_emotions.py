# This file uses a probalistic classify and count approach
import pickle
import glob
from load_filtered_tweets import BLM_TWEETS, get_date_from_filepath, pro_filter, anti_filter, cop_filter, protest_filter, blm_filter
from collections import defaultdict, Counter
from stat_utils import get_corr
import numpy as np
from config import RAW_BERT_PREDS, AGG_RAW_BERT_PREDS, EMOTIONS, CACHE_PATH
import os
import argparse
import pandas
import process_aecl
import process_ccc

LENGTH_FILTER = 5

def aggregate_results():
    tweet_id_to_emotion_scores = defaultdict(dict)
    for f in glob.iglob(RAW_BERT_PREDS):
        # preds_df = pandas.read_csv(open(f), sep="\t")
        preds = [float(x.split(",")[-1]) for x in open(f).readlines()[1:]]
        tids = f.replace(".results", ".tid")
        print(f)
        labels = [l.strip() for l in open(tids).readlines()]
        print(labels[:5])
        assert (len(preds) == len(labels)), str(len(preds)) + " " + str(len(labels))
        df = pandas.DataFrame({"preds": preds, "tid":labels})
        print(df.columns)
        print(len(df))

        emotion = f.split(".")[-2]
        def update_emo_dict(x):
            assert len(x["tid"]) == 19
            tweet_id_to_emotion_scores[x["tid"]][emotion] = x["preds"]
        df.apply(update_emo_dict, axis=1)

    pickle.dump(tweet_id_to_emotion_scores, open(AGG_RAW_BERT_PREDS, "wb"))

def cache_user_location_emotions():
    print("Caching emotions by user location")
    tweet_id_to_emotions = pickle.load(open(AGG_RAW_BERT_PREDS, "rb"))
    user_to_state = pickle.load(open(os.path.join(CACHE_PATH, "user_to_state.pkl"), "rb"))
    user_to_none = pickle.load(open(os.path.join(CACHE_PATH, "user_to_none.pkl"), "rb"))
    user_to_USA = pickle.load(open(os.path.join(CACHE_PATH, "user_to_USA.pkl"), "rb"))
    user_to_city = pickle.load(open(os.path.join(CACHE_PATH, "user_to_city.pkl"), "rb"))

    for u in user_to_USA:
        user_to_state[u] = "USA"

    for u in user_to_none:
        user_to_state[u] = "NONE"

    state_to_emotions = defaultdict(dict)
    city_to_emotions = defaultdict(dict)

    def update_dict(curr_dict, key, emotions):
        if not key in curr_dict:
            curr_dict[key]["tweet_count"] = 1
        else:
            curr_dict[key]["tweet_count"] += 1

        for emo, val in emotions.items():
            if not emo in curr_dict[key]:
                curr_dict[key][emo] = [val]
            else:
                curr_dict[key][emo].append(val)

    for f in sorted(glob.iglob(BLM_TWEETS)):
        print(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for _,tweet in tweet_dict.items():
                if len(tweet["full_text"].split()) < LENGTH_FILTER:
                    continue

                u_id = tweet['user_id']
                if u_id in user_to_state:
                    emotions = tweet_id_to_emotions[str(tweet["id"])]
                    curr_state = user_to_state[u_id]
                    update_dict(state_to_emotions, curr_state, emotions)

                    if u_id in user_to_city:
                        city_id = user_to_city[u_id] + "_" + user_to_state[u_id]
                        update_dict(city_to_emotions, city_id, emotions)

    pickle.dump(city_to_emotions, open(os.path.join(CACHE_PATH, "city_to_emotions.pcc.pickle"), "wb"))
    pickle.dump(state_to_emotions, open(os.path.join(CACHE_PATH, "state_to_emotions.pcc.pickle"), "wb"))

def view_filtered_emotions_over_time(savefilename, my_filter=None):
    tweet_id_to_emotions = pickle.load(open(AGG_RAW_BERT_PREDS, "rb"))

    date_to_emotion_counts = defaultdict(dict)
    date_to_counts = Counter()
    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        print(date)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for _,tweet in tweet_dict.items():
                if len(tweet["full_text"].split()) < LENGTH_FILTER:
                    continue
                if my_filter is not None and not my_filter(tweet):
                    continue
                date_to_counts[date] += 1
                tweet_id = str(tweet["id"])
                emotions = tweet_id_to_emotions[tweet_id]
                for e,v in emotions.items():
                    if not e in date_to_emotion_counts[date]:
                        date_to_emotion_counts[date][e] = [v]
                    else:
                        date_to_emotion_counts[date][e].append(v)

    dates = []
    date_counts = []
    emotion_counts = defaultdict(list)
    for d in sorted(date_to_emotion_counts):

        print(d.strftime("%Y/%m/%d"), end=" ")
        dates.append(d)

        print(date_to_counts[d], end=" ")
        date_counts.append(date_to_counts[d])

        for e in EMOTIONS:
            if e in ["neutral", "anger_disgust"]:
                continue
            print(np.mean(date_to_emotion_counts[d][e]), end=" ")
            emotion_counts[e].append(np.mean(date_to_emotion_counts[d][e]))
        print("")

    df_dict = {"dates": dates, "date_counts": date_counts}
    df_dict.update(emotion_counts)
    df = pandas.DataFrame.from_dict(df_dict)
    df.to_csv(savefilename)

def print_summary_stats():
    tweet_id_to_emotions = pickle.load(open(AGG_RAW_BERT_PREDS, "rb"))
    pro_to_count = defaultdict(list)
    anti_to_count = defaultdict(list)
    protests_to_count = defaultdict(list)
    cops_to_count = defaultdict(list)
    blm_to_count = defaultdict(list)
    emotions_to_count = defaultdict(list)

    skipped = 0
    missing_emotions = 0
    def do_update(my_filter, my_counter, tweet, tweet_emos):
        if my_filter is None or my_filter(tweet):
            for e,v in tweet_emos.items():
                my_counter[e].append(v)

    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        print(date)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                tokens = tweet["full_text"].split()
                if len(tokens) < LENGTH_FILTER:
                    skipped += 1
                    continue

                tweet_id = str(tweet["id"])
                if not tweet_id in tweet_id_to_emotions:
                    missing_emotions += 1
                    continue

                tweet_emotions = tweet_id_to_emotions[tweet_id]

                do_update(None, emotions_to_count, tweet, tweet_emotions)
                do_update(pro_filter, pro_to_count, tweet, tweet_emotions)
                do_update(anti_filter, anti_to_count, tweet, tweet_emotions)
                do_update(cop_filter, cops_to_count, tweet, tweet_emotions)
                do_update(protest_filter, protests_to_count, tweet, tweet_emotions)
                do_update(blm_filter, blm_to_count, tweet, tweet_emotions)

    def print_counts(my_counter, counter_name):
        print(counter_name)
        print(my_counter["tweet_count"])
        for e in sorted(my_counter):
            print(e, np.mean(my_counter[e]), len(my_counter[e]))

        percents = []
        for e in EMOTIONS:
            if e in ["neutral", "anger_disgust"]:
                continue
            percents.append(np.mean(my_counter[e]) * 100)
        return percents

    print("Skipped", skipped)
    print("Missing emotions", missing_emotions)
    all_percents = print_counts(emotions_to_count, "all")
    pro_percents = print_counts(pro_to_count, "pro-BLM")
    anti_percents = print_counts(anti_to_count, "anti-BLM")
    police_percents = print_counts(cops_to_count, "cops")
    protests_percents = print_counts(protests_to_count, "protests")
    blm_percents = print_counts(blm_to_count, "BLM")

    df_dict = {"labels": EMOTIONS,
        "all" : all_percents,
        "pro-BLM": pro_percents,
        "anti-BLM": anti_percents,
        "police": police_percents,
        "protests": protests_percents,
        "BLM-only": blm_percents}

    df = pandas.DataFrame.from_dict(df_dict)
    df.to_csv("data/keyword_split.pcc.csv")

def get_user_location_emotions(location_to_emotions, state_counter):
    missing = set()
    def get_emo_corr(e):
        protest_counts = []
        tweet_trends = []
        for s,emos in location_to_emotions.items():
            if s in state_counter:
                tweet_trends.append(np.mean(emos[e]))
                protest_counts.append(state_counter[s])
            else:
                missing.add(s)
        corrs = get_corr(tweet_trends, protest_counts)
        return corrs[0], corrs[1]

    e_to_corrs = {}
    for e in sorted(EMOTIONS):
        if not e in ["neutral", "anger_disgust"]:
            e_to_corrs[e] = get_emo_corr(e)
    print("No protest data for", missing)

    return e_to_corrs


# get city to emotions only for cities with enough users
def load_filtered_city_to_emotions(thresh=500):
    user_to_city = pickle.load(open(os.path.join(CACHE_PATH, "user_to_city.pkl"), "rb"))
    user_to_state = pickle.load(open(os.path.join(CACHE_PATH, "user_to_state.pkl"), "rb"))

    city_counts = Counter()
    for u,c in user_to_city.items():
        city = c + "_" + user_to_state[u]
        city_counts[city] += 1
    
    city_to_emotions = pickle.load(open(os.path.join(CACHE_PATH, "city_to_emotions.pcc.pickle"), "rb"))
    city_to_emotions = {l:c for l,c in city_to_emotions.items() if city_counts[l] > thresh}
    return city_to_emotions

def print_protest_corrs():
    print("################################## State corrs #################################")
    state_to_emotions = pickle.load(open(os.path.join(CACHE_PATH, "state_to_emotions.pcc.pickle"), "rb"))
    ccc_state_to_protest_count = process_ccc.get_state_to_protest_count(normalize_by_county=True)
    e_to_ccc_corrs = get_user_location_emotions(state_to_emotions, ccc_state_to_protest_count)

    aecl_state_to_protest_count = process_aecl.get_state_to_protest_count(normalize_by_county=True)
    e_to_aecl_corrs = get_user_location_emotions(state_to_emotions, aecl_state_to_protest_count)

    print("Emotion & CCC & p-val & AECL & p-val \\\\")
    for e in e_to_ccc_corrs:
        print("\\textsc{%s} & %0.2f & %0.4f & %0.2f & %0.4f \\\\" % (e, e_to_ccc_corrs[e][0], e_to_ccc_corrs[e][1], e_to_aecl_corrs[e][0], e_to_aecl_corrs[e][1]))

    print("################################## City population normalized #################################")
    city_to_emotions = load_filtered_city_to_emotions(500)
    city_to_ccc_protest_count = process_ccc.get_city_to_protest_size_normalized()
    e_to_ccc_corrs = get_user_location_emotions(city_to_emotions, city_to_ccc_protest_count)

    city_to_aecl_protest_count = process_aecl.get_city_to_protest_size_normalized()
    e_to_aecl_corrs = get_user_location_emotions(city_to_emotions, city_to_aecl_protest_count)

    print("Emotion & CCC & p-val & AECL & p-val \\\\")
    for e in e_to_ccc_corrs:
        print("\\textsc{%s} & %0.2f & %0.4f & %0.2f & %0.4f \\\\" % (e, e_to_ccc_corrs[e][0], e_to_ccc_corrs[e][1], e_to_aecl_corrs[e][0], e_to_aecl_corrs[e][1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--over_time", action='store_true')
    parser.add_argument("--refresh_cache", action='store_true')
    parser.add_argument("--filtered_over_time", action='store_true')
    parser.add_argument("--print_summary_stats", action='store_true')
    parser.add_argument("--print_location_corrs", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(AGG_RAW_BERT_PREDS):
        aggregate_results()

    if not os.path.exists(os.path.join(CACHE_PATH, "city_to_emotions.pcc.pickle")) or args.refresh_cache:
        cache_user_location_emotions()

    if args.over_time:
        view_filtered_emotions_over_time("data/emotions_over_time.pcc.csv")

    if args.filtered_over_time:
        view_filtered_emotions_over_time("data/pro_emotions_over_time.pcc.csv", my_filter=pro_filter)
        view_filtered_emotions_over_time("data/anti_emotions_over_time.pcc.csv", my_filter=anti_filter)

    if args.print_summary_stats:
        print_summary_stats()

    if args.print_location_corrs:
        print_protest_corrs()

if __name__ == "__main__":
    # tweet_id_to_emotions = pickle.load(open(AGG_RAW_BERT_PREDS, "rb"))
    # print(len(tweet_id_to_emotions))
    # aggregate_results()
    main()
