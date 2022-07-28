############################# NEED TO FILL IN ######################################################################
# Path to a local directory to store files
CACHE_PATH = ""


# Path to tweet directory. This is is a assumed to be a glob of files (e.g. "~/blm/*.pkl"), where each file is named according
# to a date e.g. 06-06-original.pkl, and each file contains all the tweets gathered for that date. The expected format is a dictionary of
# {tweet_id:tweet_info} and the dict is expected to be saved using pickle.
# tweet_info should contain the tweet full text and the list of search terms the tweet contains. The full list of search terms can be found in 
# load_filtered_tweets.py
# The function load_filtered_tweets/get_date_from_filepath can be editted according to file naming convention
# BLM_TWEETS = ""
BLM_TWEETS = "data/*-original.pkl" # this drops the \#NeverForget1984 tweets


# Path to cached outputs of emotion style (ekman emotions). Format is expected to be a dict of {tweet_id:[emotions in tweet]}, e.g.
# {1:['anger','disgust']} saved using pickle
# EMOTION_PATH = ""
EMOTION_PATH = "res/tid_emotions/tid_emotions_binary_wfear_new.pkl" # Includes fear


# Path to cached user location strs. Formated is expected to be {user_id:"user-populated location string"} saved using .pkl.gz
# This is used by process_user_locations.py and is only needed for geographic correlations with on-the-ground protests
# if you don't have this file you should comment out the cache_user_location_emotions lines in process_emotions.py
user_to_location_str = ""

# Paths to cached 2014 data, only needed if separately running 2014 data
EMOTION_2014_PATH = ""
TWEET_2014_GLOB = ""

# Paths to raw predictions scores outputted by classification model. This is used for PCC in the appendix, not needed for main results
RAW_BERT_PREDS = ""
AGG_RAW_BERT_PREDS = ""

# Paths to cached sent140 and VADER outputs. Used for appendix results, not needed for main results
SENT_140_all = ""
SENT_140_agg = ""

VADER_all = ""
VADER_agg = ""
#################################################################################################################################

EXTRA_DATA_DIR = "./external_data" # Directory containing external data like ACELD and hand-processed city names
EMOTIONS = ['anger','disgust','joy','surprise','fear','sadness', 'neutral']# 'anger_disgust'] # newer cache doesn't include anger/disgust
SENTIMENT = ['positive','negative','neutral']


def get_filepaths(key):
    if key == "sentiment140":
        return SENT_140_all, SENT_140_agg, SENTIMENT

    if key == "vader":
        return VADER_all, VADER_agg, SENTIMENT

    if key == "ekman":
        return None, EMOTION_PATH, EMOTIONS

    assert False, "Unknown key" + key

