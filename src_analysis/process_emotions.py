# This file uses a "classify and count" approach
import pickle
import glob
from load_filtered_tweets import BLM_TWEETS, get_date_from_filepath, pro_filter, anti_filter, cop_filter, protest_filter, blm_filter
from collections import defaultdict, Counter
import datetime
from stat_utils import make_percent_change, average, make_series, get_corr
import numpy as np
import math
from config import CACHE_PATH, EMOTION_2014_PATH, TWEET_2014_GLOB, get_filepaths
import os
import argparse
from basic_log_odds import write_log_odds, print_polar_words
import json
import pandas
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import process_aecl
import process_ccc

LENGTH_FILTER = 5

def make_word_cloud(word_freq, filename):
    word_freq = {w:s for w,s in word_freq.items() if not w in STOPWORDS}
    # wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    wordcloud = WordCloud().generate_from_frequencies(word_freq)
    wordcloud.to_file(filename)

def find_pro_blm_users():
    user_to_count = Counter()
    for f in sorted(glob.iglob(BLM_TWEETS)):
        print(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                if not pro_filter(tweet):
                    continue
                user_to_count[tweet['user_id']] += 1
    print("Number of users", len(user_to_count))
    pickle.dump(user_to_count, open("user_to_proBLM_count.pkl", "wb"))
    user_to_count = {u:c for u,c in user_to_count.items() if c > 5}
    print("Number of filtered users", len(user_to_count))


def cache_user_location_emotions(key, my_filter = None, filter_str= ""):
    _, emotion_cache_path, labels = get_filepaths(key)
    print("Caching emotions by user location")
    tweet_id_to_emotions = pickle.load(open(emotion_cache_path, "rb"))
    user_to_state = pickle.load(open(os.path.join(CACHE_PATH, "user_to_state.pkl"), "rb"))
    user_to_none = pickle.load(open(os.path.join(CACHE_PATH, "user_to_none.pkl"), "rb"))
    user_to_USA = pickle.load(open(os.path.join(CACHE_PATH, "user_to_USA.pkl"), "rb"))
    user_to_city = pickle.load(open(os.path.join(CACHE_PATH, "user_to_city.pkl"), "rb"))

    for u in user_to_USA:
        user_to_state[u] = "USA"

    for u in user_to_none:
        user_to_state[u] = "NONE"

    state_to_emotions = defaultdict(Counter)
    city_to_emotions = defaultdict(Counter)

    state_to_date_to_emotions = defaultdict(dict)
    city_to_date_to_emotions = defaultdict(dict)

    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                if len(tweet["full_text"].split()) < LENGTH_FILTER:
                    continue

                if my_filter is not None and not my_filter(tweet):
                    continue

                # if "#blacklivesmatter" in tweet["search_expressions"]:
                #     continue

                u_id = tweet['user_id']
                if u_id in user_to_state:
                    emotions = tweet_id_to_emotions[str(tweet["id"])]
                    curr_state = user_to_state[u_id]
                    state_to_emotions[user_to_state[u_id]].update(emotions)
                    state_to_emotions[user_to_state[u_id]]["tweet_count"] += 1

                    if date in state_to_date_to_emotions[curr_state]:
                        state_to_date_to_emotions[curr_state][date].update(emotions)
                    else:
                        state_to_date_to_emotions[curr_state][date] = Counter(emotions)
                    state_to_date_to_emotions[curr_state][date]["tweet_count"] += 1

                    if u_id in user_to_city:
                        city_id = user_to_city[u_id] + "_" + user_to_state[u_id]
                        city_to_emotions[city_id].update(emotions)
                        city_to_emotions[city_id]["tweet_count"] += 1

                        if date in city_to_date_to_emotions[city_id]:
                            city_to_date_to_emotions[city_id][date].update(emotions)
                        else:
                            city_to_date_to_emotions[city_id][date] = Counter(emotions)
                        city_to_date_to_emotions[city_id][date]["tweet_count"] += 1

    user_to_percent_change = {}
    print(len(city_to_emotions), len(state_to_emotions))
    # pickle.dump(city_to_emotions, open(os.path.join(CACHE_PATH, "city_to_emotions.pickle"), "wb"))
    # pickle.dump(state_to_emotions, open(os.path.join(CACHE_PATH, "state_to_emotions.pickle"), "wb"))
    # pickle.dump(city_to_date_to_emotions, open(os.path.join(CACHE_PATH, "city_to_date_to_emotions.pickle"), "wb"))
    # pickle.dump(state_to_date_to_emotions, open(os.path.join(CACHE_PATH, "state_to_date_to_emotions.pickle"), "wb"))

    # cache = "/usr1/home/anjalief/blm_2020/skip_blm_cache/"
    cache = CACHE_PATH
    pickle.dump(city_to_emotions, open(os.path.join(cache, "city_to_emotions.%s%s.pickle" % (key, filter_str)), "wb"))
    pickle.dump(state_to_emotions, open(os.path.join(cache, "state_to_emotions.%s%s.pickle" % (key, filter_str)), "wb"))
    pickle.dump(city_to_date_to_emotions, open(os.path.join(cache, "city_to_date_to_emotions.%s%s.pickle" % (key, filter_str)), "wb"))
    pickle.dump(state_to_date_to_emotions, open(os.path.join(cache, "state_to_date_to_emotions.%s%s.pickle" % (key, filter_str)), "wb"))


def print_emotion_retweets_and_hashtags(key, my_filter):
    _, emotion_cache_path, labels = get_filepaths(key)

    tweet_id_to_emotions = pickle.load(open(emotion_cache_path, "rb"))
    emotion_to_retweet_count = defaultdict(list)
    emotion_to_hashtag_counts = defaultdict(Counter)
    all_hashtag_counts = Counter()
    all_word_counts = Counter()
    emotion_to_word_counts = defaultdict(Counter)

    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        print(date)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                tokens = tweet["full_text"].split()
                if len(tokens) < LENGTH_FILTER:
                    continue
                if my_filter is not None and not my_filter(tweet):
                    continue
                tweet_id = str(tweet["id"])
                tweet_emotions = tweet_id_to_emotions[tweet_id]

                tokens = [t for t in tokens if not t.startswith("@")]
                hashtags = [h['text'] for h in tweet["entities"]['hashtags']]
                all_hashtag_counts.update(hashtags)
                all_word_counts.update(tokens)
                for te in tweet_emotions:
                    emotion_to_retweet_count[te].append(tweet["retweet_count"])
                    emotion_to_hashtag_counts[te].update(hashtags)
                    emotion_to_word_counts[te].update(tokens)
                    if te not in labels:
                        print(tweet_emotions, i, tweet_id)
    print("emotion mean min max std")
    for e,retweets in emotion_to_retweet_count.items():
        print(e, np.mean(retweets), np.min(retweets), np.max(retweets), np.std(retweets))

    for e, hashtag_counts in emotion_to_hashtag_counts.items():
        print("#######################################", e, "###########################################################")
        delta = write_log_odds(hashtag_counts, all_hashtag_counts, all_hashtag_counts)
        print_polar_words(delta, 20)

    if my_filter is None:
        for e, word_counts in emotion_to_word_counts.items():
            delta = write_log_odds(word_counts, all_word_counts, all_word_counts)
            make_word_cloud(delta, "plots/" + e + "_odds_word_cloud.%s.png" % key)

def print_user_topic_correlations(key):
    _, emotion_path, emotion_labels = get_filepaths
    tweet_id_to_emotions = pickle.load(open(emotion_path, "rb"))
    user_id_to_emotions = defaultdict(Counter)
    user_id_to_search_terms = defaultdict(Counter)
    user_id_to_count = Counter()

    pro_to_count = Counter()
    anti_to_count = Counter()
    protests_to_count = Counter()
    cops_to_count = Counter()
    emotions_to_count = Counter()

    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                if len(tweet["full_text"].split()) < LENGTH_FILTER:
                    continue
                tweet_id = str(tweet["id"])
                tweet_emotions = tweet_id_to_emotions[tweet_id]
                user_id = tweet["user_id"]
                
                user_id_to_count[user_id] += 1

                for te in tweet_emotions:
                    user_id_to_emotions[user_id][te] += 1

                if pro_filter(tweet):
                    user_id_to_search_terms[user_id]["PRO"] += 1
                #     if not "anger" in tweet_emotions:
                #         user_id_to_search_terms[user_id]["PRO_no_anger"] += 1
                #     if not "joy" in tweet_emotions:
                #         user_id_to_search_terms[user_id]["PRO_no_joy"] += 1
                # else:
                #     if "anger" in tweet_emotions:
                #         user_id_to_emotions[user_id]["anger_no_PRO"] += 1
                #     if "joy" in tweet_emotions:
                #         user_id_to_emotions[user_id]["joy_no_PRO"] += 1



                if anti_filter(tweet):
                    user_id_to_search_terms[user_id]["ANTI"] += 1

                if cop_filter(tweet):
                    user_id_to_search_terms[user_id]["COPS"] += 1
                #     if not "anger" in tweet_emotions:
                #         user_id_to_search_terms[user_id]["COPS_no_anger"] += 1
                #     if not "joy" in tweet_emotions:
                #         user_id_to_search_terms[user_id]["COPS_no_joy"] += 1
                # else:
                #     if "anger" in tweet_emotions:
                #         user_id_to_emotions[user_id]["anger_no_COPS"] += 1
                #     if "joy" in tweet_emotions:
                #         user_id_to_emotions[user_id]["joy_no_COPS"] += 1



                if protest_filter(tweet):
                    user_id_to_search_terms[user_id]["PROTEST"] += 1

    # Only keep users with at least 10 tweets
    user_id_to_count = {u:c for u,c in user_id_to_count.items() if c > 10}
    print("Number of users included", len(user_id_to_count))
    user_id_to_emotions = user_id_to_emotions
    user_id_to_search_terms = user_id_to_search_terms

    emotions_to_vals = defaultdict(list)
    search_terms_to_vals = defaultdict(list)
    filter_keys = ["PRO", "ANTI", "COPS", "PROTEST"] #"COPS_no_anger", "COPS_no_joy", "PRO_no_anger", "PRO_no_joy"]
    for u,tweet_count in user_id_to_count.items():
        for e in emotion_labels: # + ["anger_no_COPS", "joy_no_COPS", "anger_no_PRO", "joy_no_PRO"]:
            emotions_to_vals[e].append(user_id_to_emotions[u][e] / tweet_count)

        for k in filter_keys:
            search_terms_to_vals[k].append(user_id_to_search_terms[u][k] / tweet_count)

    for e,e_vals in emotions_to_vals.items():
        for s,s_vals in search_terms_to_vals.items():
            print(e, s, get_corr(e_vals, s_vals))


def print_tweet_topic_correlations(key, top_users_only = False):
    _, emotion_path, emotion_labels = get_filepaths(key)
    tweet_id_to_emotions = pickle.load(open(emotion_path, "rb"))
    emotions_to_vals = defaultdict(list)
    search_terms_to_vals = defaultdict(list)

    key_to_filter = {"ANTI": anti_filter,
    "PRO": pro_filter,
    "COPS": cop_filter,
    "PROTESTS": protest_filter
    }
    user_id_to_count = Counter()
    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                user_id_to_count[tweet["user_id"]] += 1

    user_id_to_count = {u:c for u,c in user_id_to_count.items() if c > 10}
    print("Keeping users:", len(user_id_to_count))
    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                if len(tweet["full_text"].split()) < LENGTH_FILTER:
                    continue

                if top_users_only and not tweet["user_id"] in user_id_to_count:
                    continue

                tweet_id = str(tweet["id"])
                tweet_emotions = tweet_id_to_emotions[tweet_id]

                for e in emotion_labels:
                    if e in tweet_emotions:
                        emotions_to_vals[e].append(1)
                    else:
                        emotions_to_vals[e].append(0)

                for k,my_filter in key_to_filter.items():
                    if my_filter(tweet):
                        search_terms_to_vals[k].append(1)
                    else:
                        search_terms_to_vals[k].append(0)

    print("Number of tweets", len(emotions_to_vals["anger"]))
    for e in emotion_labels:
        for k in key_to_filter:
            print(e, k, get_corr(emotions_to_vals[e], search_terms_to_vals[k]))


def get_user_location_emotions(location_to_emotions, state_counter, print_all, key, verbose = False):
    _, _, labels = get_filepaths(key)

    tweet_counts = 0
    for s,e_counts in location_to_emotions.items():
        tweet_counts += e_counts["tweet_count"]
    print("Total tweet count", tweet_counts)


    if print_all:
        for s,e_counts in location_to_emotions.items():
            if e_counts["tweet_count"] == 0:
                print(s, ",0")
                continue
            print(s, end=",")
            for e in sorted(labels):
                print(e_counts[e] / e_counts["tweet_count"], end=",")
            print(e_counts["tweet_count"])

    missing = set()

    def get_emo_corr(e, normalize):
        protest_counts = []
        tweet_trends = []
        for s,emos in location_to_emotions.items():
            if s in state_counter: # and state_counter[s] > 3:
                if normalize:
                    tweet_trends.append(emos[e] / emos["tweet_count"])
                else:
                    tweet_trends.append(emos[e])
                protest_counts.append(state_counter[s])
            else:
                missing.add(s)
        corrs = get_corr(tweet_trends, protest_counts)
        if verbose:
            print("Number included", len(tweet_trends))
        return corrs[0], corrs[1]

    e_to_corrs = {}
    for e in sorted(labels):
        e_to_corrs[e] = get_emo_corr(e, True)
    if verbose:
        print("No protest data for", missing)

    return e_to_corrs

def print_summary_stats(key):
    _, emotion_cache_path, labels = get_filepaths(key)
    tweet_id_to_emotions = pickle.load(open(emotion_cache_path, "rb"))
    pro_to_count = Counter()
    anti_to_count = Counter()
    neutral_to_count = Counter()
    protests_to_count = Counter()
    cops_to_count = Counter()
    blm_to_count = Counter()
    emotions_to_count = Counter()
    emotions_to_word_counts = {e:Counter() for e in labels}

    skipped = 0
    def do_update(my_filter, my_counter, tweet, tweet_emos):
        if my_filter is None or my_filter(tweet):
            my_counter.update(tweet_emos)
            my_counter["tweet_count"] += 1

    print_count = 0
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
                tweet_emotions = tweet_id_to_emotions[tweet_id]

                for te in tweet_emotions:
                    if te not in labels:
                        print(tweet_emotions, i, tweet_id)
                    else:
                        tokens = [t for t in tokens if not t.startswith("@")]
                        emotions_to_word_counts[te].update(tokens)

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
            print(e, my_counter[e] / my_counter["tweet_count"])

        percents = []
        for e in labels:
            percents.append(my_counter[e] / my_counter["tweet_count"] * 100)
        return percents

    print(emotions_to_count)
    print("Skipped", skipped)
    all_percents = print_counts(emotions_to_count, "all")
    pro_percents = print_counts(pro_to_count, "pro-BLM")
    anti_percents = print_counts(anti_to_count, "anti-BLM")
    police_percents = print_counts(cops_to_count, "cops")
    protests_percents = print_counts(protests_to_count, "protests")
    blm_percents = print_counts(blm_to_count, "BLM")

    df_dict = {"labels": labels,
        "all" : all_percents,
        "pro-BLM": pro_percents,
        "anti-BLM": anti_percents,
        "police": police_percents,
        "protests": protests_percents,
        "BLM-only": blm_percents}

    df = pandas.DataFrame.from_dict(df_dict)
    df.to_csv("data/keyword_split.%s.csv" % key)


    print(STOPWORDS)
    for e in labels:
        if e == "neutral":
            continue
        make_word_cloud(emotions_to_word_counts[e], "plots/" + e + "_word_cloud." + key + ".png")

def view_filtered_emotions_over_time(key, savefilename, my_filter=None):
    _, emotion_cache_path, labels = get_filepaths(key)
    tweet_id_to_emotions = pickle.load(open(emotion_cache_path, "rb"))

    date_to_emotion_counts = defaultdict(Counter)
    date_to_counts = Counter()
    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        print(date)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                if len(tweet["full_text"].split()) < LENGTH_FILTER:
                    continue
                if my_filter is not None and not my_filter(tweet):
                    continue
                date_to_counts[date] += 1
                tweet_id = str(tweet["id"])
                date_to_emotion_counts[date].update(tweet_id_to_emotions[tweet_id])
    for e in labels:
        print(e, end=" ")
    print("")
    dates = []
    date_counts = []
    emotion_counts = {e:[] for e in labels}
    for d in sorted(date_to_emotion_counts):

        print(d.strftime("%Y/%m/%d"), end=" ")
        dates.append(d)

        print(date_to_counts[d], end=" ")
        date_counts.append(date_to_counts[d])

        for e in labels:
            print(date_to_emotion_counts[d][e] / date_to_counts[d], end=" ")
            emotion_counts[e].append(date_to_emotion_counts[d][e])
        print("")

    df_dict = {"dates": dates, "date_counts": date_counts}
    df_dict.update(emotion_counts)
    df = pandas.DataFrame.from_dict(df_dict)
    df.to_csv(savefilename)
    # pickle.dump((date_to_counts, date_to_emotion_counts), open("date_to_filtered_emotions.pkl", "wb"))

def fix_cities():
    user_to_city = pickle.load(open(os.path.join(CACHE_PATH, "user_to_city.pickle"), "rb"))
    p_fixed = 0
    t_fixed = 0

    for u, city in user_to_city.items():
        if city == "Pheonix":
            user_to_city[u] = "Phoenix"
            p_fixed += 1
        if city == "Tuscon":
            user_to_city[u] = "Tucson"
            t_fixed += 1
    print(p_fixed, t_fixed)
    pickle.dump(user_to_city, open(os.path.join(CACHE_PATH, "user_to_city.pickle"), "wb"))

# get city to emotions only for cities with enough users
def load_filtered_city_to_emotions(key, filter_str, thresh=500, print_state_counts=False):
    user_to_city = pickle.load(open(os.path.join(CACHE_PATH, "user_to_city.pkl"), "rb"))
    user_to_state = pickle.load(open(os.path.join(CACHE_PATH, "user_to_state.pkl"), "rb"))

    if print_state_counts:
        state_to_user_counts = Counter(list(user_to_state.values()))
        for s,u in state_to_user_counts.items():
            print("%s,%s" % (s,u))

    city_counts = Counter()
    for u,c in user_to_city.items():
        city = c + "_" + user_to_state[u]
        city_counts[city] += 1
    
    city_to_emotions = pickle.load(open(os.path.join(CACHE_PATH, "city_to_emotions.%s%s.pickle" % (key, filter_str)), "rb"))
    city_to_emotions = {l:c for l,c in city_to_emotions.items() if city_counts[l] > thresh}
    return city_to_emotions

def print_2014_overview():
    tweet_id_to_emotions = pickle.load(open(EMOTION_2014_PATH, 'rb'))

    skipped = 0
    print_count = 0
    emotion_to_hashtag_counts = defaultdict(Counter)
    overall_emotion_counts = Counter()
    all_hashtag_counts = Counter()
    for filename in glob.iglob(TWEET_2014_GLOB):
        tweets = json.load(open(filename))
        for t in tweets:
            splits = t["full_text"].split()
            if len(splits) < 5:
                skipped += 1
                continue
            hashtags = [w for w in splits if w.startswith("#")]
            all_hashtag_counts.update(hashtags)
            emotions = tweet_id_to_emotions[str(t['id'])]
            if print_count < 50:
                print(t["full_text"], emotions)
                print_count += 1

            for e in emotions:
                emotion_to_hashtag_counts[e].update(hashtags)
            overall_emotion_counts.update(emotions)

    print("Skipped", skipped)
    total = len(tweet_id_to_emotions) - skipped
    print(total)
    for e in overall_emotion_counts:
        print(e, overall_emotion_counts[e]/ total)

    for e, hashtag_counts in emotion_to_hashtag_counts.items():
        print("#######################################", e, "###########################################################")
        delta = write_log_odds(hashtag_counts, all_hashtag_counts, all_hashtag_counts)
        print_polar_words(delta, 20)

def compute_emotion_correlations(key):
    _, emotion_cache_path, _ = get_filepaths(key)
    tweet_id_to_emotions = pickle.load(open(emotion_cache_path, "rb"))
    anger = []
    joy = []
    disgust = []

    def update_count(emos, emo_str, emo_list):
        if emo_str in emos:
            emo_list.append(1)
        else:
            emo_list.append(0)

    for i,emotions in tweet_id_to_emotions.items():
        update_count(emotions, "anger", anger)
        update_count(emotions, "disgust", disgust)
        update_count(emotions, "joy", joy)
    print("Anger vs. joy", get_corr(anger, joy))
    print("Disgust vs. joy", get_corr(disgust, joy))
    print("Anger vs. disgust", get_corr(anger, disgust))

def print_protest_corrs(key, filter_str):
    _, _, labels = get_filepaths(key)
    print("################################## State corrs %s #################################" % filter_str)
    state_to_emotions = pickle.load(open(os.path.join(CACHE_PATH, "state_to_emotions.%s%s.pickle" % (key, filter_str)), "rb"))
    ccc_state_to_protest_count = process_ccc.get_state_to_protest_count(normalize_by_county=True)
    e_to_ccc_corrs = get_user_location_emotions(state_to_emotions, ccc_state_to_protest_count, False, key)

    aecl_state_to_protest_count = process_aecl.get_state_to_protest_count(normalize_by_county=True)
    e_to_aecl_corrs = get_user_location_emotions(state_to_emotions, aecl_state_to_protest_count, False, key)

    print("Emotion & CCC & p-val & AECL & p-val \\\\")
    for e in labels:
        print("\\textsc{%s} & %0.2f & %0.4f & %0.2f & %0.4f \\\\" % (e, e_to_ccc_corrs[e][0], e_to_ccc_corrs[e][1], e_to_aecl_corrs[e][0], e_to_aecl_corrs[e][1]))

    city_to_emotions = load_filtered_city_to_emotions(key, filter_str, 500, False)

    print("################################## City population normalized #################################")
    city_to_ccc_protest_count = process_ccc.get_city_to_protest_size_normalized()
    e_to_ccc_corrs = get_user_location_emotions(city_to_emotions, city_to_ccc_protest_count, False, key)

    city_to_aecl_protest_count = process_aecl.get_city_to_protest_size_normalized()
    e_to_aecl_corrs = get_user_location_emotions(city_to_emotions, city_to_aecl_protest_count, False, key)

    print("Emotion & CCC & p-val & AECL & p-val \\\\")
    for e in labels:
        print("\\textsc{%s} & %0.2f & %0.4f & %0.2f & %0.4f \\\\" % (e, e_to_ccc_corrs[e][0], e_to_ccc_corrs[e][1], e_to_aecl_corrs[e][0], e_to_aecl_corrs[e][1]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", choices=['ekman', 'vader', "sentiment140"], help="ekman is the primary data in the paper. vader and sentiment140 models are used in the appendix")
    parser.add_argument("--over_time", action='store_true')
    parser.add_argument("--refresh_cache", action='store_true')
    parser.add_argument("--filtered_over_time", action='store_true')
    parser.add_argument("--print_summary_stats", action='store_true')
    parser.add_argument("--print_location_corrs", action='store_true')
    parser.add_argument("--print_user_topic_corr", action='store_true')
    parser.add_argument("--retweets_and_hashtags", action='store_true')
    parser.add_argument("--print_2014", action='store_true')
    parser.add_argument("--print_emo_corrs", action='store_true')
    args = parser.parse_args()

    # if not os.path.exists(os.path.join(CACHE_PATH, "city_to_emotions." + args.key + ".pickle")) or args.refresh_cache:
    #     cache_user_location_emotions(args.key)
    #     cache_user_location_emotions(args.key, pro_filter, ".pro")
    #     cache_user_location_emotions(args.key, anti_filter, ".anti")

    if args.over_time:
        view_filtered_emotions_over_time(args.key, "data/emotions_over_time." + args.key + ".csv")

    if args.filtered_over_time:
        view_filtered_emotions_over_time(args.key, "data/pro_emotions_over_time." + args.key + ".csv", my_filter=pro_filter)
        view_filtered_emotions_over_time(args.key, "data/anti_emotions_over_time." + args.key + ".csv", my_filter=anti_filter)

    if args.print_summary_stats:
        print_summary_stats(args.key)

    if args.print_location_corrs:
        print_protest_corrs(args.key, "")
        print_protest_corrs(args.key, ".pro")
        print_protest_corrs(args.key, ".anti")

    if args.retweets_and_hashtags:
        print("No filter")
        print_emotion_retweets_and_hashtags(args.key, my_filter=None)
        print("Pro filter")
        print_emotion_retweets_and_hashtags(args.key, my_filter=pro_filter)
        print("Anti filter")
        print_emotion_retweets_and_hashtags(args.key, my_filter=anti_filter)


    if args.print_user_topic_corr:
        print_user_topic_correlations(args.key)
        print_tweet_topic_correlations(args.key, True)
        print_tweet_topic_correlations(args.key)

    if args.print_2014:
        print_2014_overview()

    if args.print_emo_corrs:
        compute_emotion_correlations(args.key)

if __name__ == "__main__":
    main()
