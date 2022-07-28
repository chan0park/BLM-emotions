import argparse
import glob
import pickle
from collections import Counter
import os
import datetime
import random
from process_aecl import load_relevant_aecl_data, get_events_per_city
from config import BLM_TWEETS, CACHE_PATH

# On our original list but not in the data:
# #besafeoutthere 0
# #policebrutality 0
# georgefloyd 0
# #altright 0
# #georgefloydwasmurdered 0

# Counts over full data set
# Counter({'police': 11543574, 'cops': 4540549, '#blacklivesmatter': 4464081, 'protests': 3690422, 'george floyd': 2817319, 'protest': 2616210, 'protesters': 2077581, 'riots': 1658971, '#blm': 1070460, 'looting': 1037082, '#georgefloyd': 963898, 'protestors': 845071, 'riot': 794793, 'rioters': 598690, 'looters': 466232, '#justiceforgeorgefloyd': 395882, '#alllivesmatter': 228541, '#blackouttuesday': 207201, '#georgefloydprotests': 147841, '#icantbreathe': 145104, 'derek chauvin': 141844, '#nojusticenopeace': 124539, '#bluelivesmatter': 95800, '#whitelivesmatter': 86434, '#justiceforfloyd': 58362, '#alllivesmatters': 46212, '#voteouthate': 41846, '#alllivesmattter': 41790, '#theshowmustbepaused': 18175, '#whitelivesmatters': 13658, '#neverforget1984': 11993, '#whitelifematters': 9909, '#whitelifematter': 6152, '#blackwomenmatter': 5278, '#walkwithus': 3025, '#whitelivesmattermore': 2213, '#whitelivesmattermost': 1238, '#blackgirlsmatter': 767, '#kneelwithus': 573, '#worldagainstracism': 394})

all_search_terms = ["george floyd", "protest", "protests", "protesters", "protestors", "riots", "riot", "rioters", "looting", "looters", "cops", "police",
    "#BlackLivesMatter", "#ICantBreathe", "#BLM", "#JusticeForFloyd", "#JusticeForGeorgeFloyd", "#GeorgeFloydProtests",
    "#WorldAgainstRacism", "#WalkWithUs", "#KneelWithUs", "#blacklivesmatter", "#georgefloyd", "#blm", "#justiceforfloyd", "#BlackoutTuesday",
    "#neverforget1984", "#theshowmustbepaused", "#VoteOutHate", "#NoJusticeNoPeace", "#BlackWomenMatter", "#BlackGirlsMatter",
    "#BlueLivesMatter", "#AllLivesMatter",
    "#AltRight", "#GeorgeFloydProtests",
    "#AllLivesMattter", "#WhiteLifeMatters", "#WhiteLivesMatter", "#WhiteLifeMatter", "#WhiteLivesMatters", "#WhiteLivesMatterMost",
     "#whitelivesmattermore", "derek chauvin"]
all_search_terms = set([i.lower() for i in all_search_terms])

anti_search_terms = ["#BlueLivesMatter", "#AllLivesMatter", "#alllivesmatters", "#AllLivesMattter",
    "#WhiteLifeMatters", "#WhiteLivesMatter", "#WhiteLifeMatter", "#WhiteLivesMatters", "#WhiteLivesMatterMost",
    "#whitelivesmattermore"]
anti_search_terms = set([i.lower() for i in anti_search_terms])

pro_search_terms = ["#BlackLivesMatter", "#ICantBreathe", "#BLM", "#JusticeForFloyd", "#JusticeForGeorgeFloyd", "#GeorgeFloydProtests",
    "#WorldAgainstRacism", "#WalkWithUs", "#KneelWithUs", "#blacklivesmatter", "#georgefloyd", "#blm", "#justiceforfloyd", "#BlackoutTuesday",
    "#theshowmustbepaused", "#VoteOutHate", "#NoJusticeNoPeace", "#BlackWomenMatter", "#BlackGirlsMatter",
    "#GeorgeFloydProtests"]
pro_search_terms = set([i.lower() for i in pro_search_terms])


city_to_terms = {
"Portland": "Portland",
"Richmond": "Richmond",
"Seattle": "Seattle",
"Columbus": "Columbus",
"Columbia": "Columbia",
"Los Angeles": "Los Angeles",
"Atlanta": "Atlanta",
"New York-Manhattan": "Manhattan",
"Detroit": "Detroit",
"Chicago": "Chicago",
"Phoenix": "Phoenix",
"Springfield": "Springfield",
"New York-Brooklyn": "Brooklyn",
"Des Moines": "Des Moines",
"Memphis": "Memphis",
"Denver": "Denver",
"Louisville": "Louisville",
"Oakland": "Oakland",
"Milwaukee": "Milwaukee",
"Washington DC": "Washington DC",
"Seattle-CHOP": "Seattle-CHOP",
"Boston": "Boston",
"Raleigh": "Raleigh",
"Jacksonville": "Jacksonville",
"Kansas City": "Kansas City",
"Madison": "Madison",
"Minneapolis": "Minneapolis",
"San Francisco": "San Francisco",
"Lexington": "Lexington",
"Pittsburgh": "Pittsburgh",
"San Diego": "San Diego",
"Miami": "Miami",
"San Antonio": "San Antonio",
"Philadelphia": "Philadelphia",
"Las Vegas": "Las Vegas",
"Jackson": "Jackson",
"Wilmington": "Wilmington",
"Charleston": "Charleston",
"Chattanooga": "Chattanooga",
"Philadelphia": "Philadelphia",
"New York-Queens": "Queens",
"Lincoln":"Lincoln",
"Salt Lake City": "Salt Lake City",
"Dallas":"Dallas",
"Indianapolis":"Indianapolis",
"Eugene":"Eugene",
"New Orleans":"New Orleans",
"Athens":"Athens",
"Austin":"Austin",
"LaFayette":"LaFayette",
"Saint Paul":"Saint Paul",
"Charlotte":"Charlotte",
"Arlington":"Arlington",
"Iowa City":"Iowa City",
"Tampa":"Tampa",
"Sacramento":"Sacramento",
"Tulsa":"Tulsa",
"Albuquerque":"Albuquerque",
"Tallahassee":"Tallahassee",
"Baltimore":"Baltimore",
"Cleveland":"Cleveland",
"Winston-Salem":"Winston-Salem",
"Saint Petersburg":"Saint Petersburg",
"Laramie":"Laramie",
"Birmingham":"Birmingham",
"Colorado Springs":"Colorado Springs",
"Rochester":"Rochester",
"New York-Bronx":"New York-Bronx",
"Norfolk":"Norfolk"
}

kpop_terms = ["kpop", "bts", "stan ", "bias", "fan cam"]

def pro_filter(tweet):
    if not any([x in pro_search_terms for x in  tweet['search_expressions']]):
        return False
    if any([x in anti_search_terms for x in tweet['search_expressions']]):
        return False
    return True

def anti_filter(tweet):
    if not any([x in anti_search_terms for x in tweet['search_expressions']]):
        return False
    if any([x in pro_search_terms for x in tweet['search_expressions']]):
        return False
    return True

def cop_filter(tweet):
    if "cops" in tweet['search_expressions'] or "police" in tweet["search_expressions"]:
        return True
    return False

def blm_filter(tweet):
    if "#blacklivesmatter" in tweet["search_expressions"]:
        return True
    return False


def protest_filter(tweet):
    if "protests" in tweet["search_expressions"] or \
        "protest" in tweet["search_expressions"] or \
        "protesters" in tweet["search_expressions"] or \
        "protestors" in tweet["search_expressions"]:
        return True
    return False


def get_date_from_filepath(f):
    basename = os.path.basename(f)
    date_str = basename.split("-original")[0]
    parts = date_str.split("-")
    date = datetime.datetime(year=2020, month=int(parts[0]), day=int(parts[1]))
    return date

def get_sample_tweets_per_city():
    print("Running with pro-filter")
    city_to_counts_per_day = {}
    city_to_counts = Counter()
    city_to_samples = {}
    city_to_ids = {}
    aecl_data = load_relevant_aecl_data()
    city_to_event_count = get_events_per_city(aecl_data)
    city_to_event_count = {city_to_terms.get(c, c).lower():v for c,v in city_to_event_count.items()}
    for c in city_to_event_count:
        city_to_counts_per_day[c] = Counter()
        city_to_samples[c] = []
        city_to_ids[c] = []

    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                if not any([x in pro_search_terms for x in  tweet['search_expressions']]):
                    continue
                if any([x in anti_search_terms for x in tweet['search_expressions']]):
                    continue
                text = tweet['full_text'].lower()
                cities_to_update = [city for city in city_to_event_count if city in text or city.replace(" ", "") in text]
                # Some tweets just list all cities where protests are happening, not what we want
                if len(cities_to_update) > 2:
                    continue
                for city in cities_to_update:
                    city_to_counts_per_day[city][date] += 1
                    city_to_counts[city] += 1
                    # get rid of newlines cause they print ugly
                    city_to_samples[city].append(" ".join(tweet["full_text"].split()))
                    city_to_ids[city].append(tweet)
    pickle.dump(city_to_samples, open(os.path.join(CACHE_PATH, "cache_city_tweets_samples_pro2.pickle"), "wb"))
    pickle.dump(city_to_ids, open(os.path.join(CACHE_PATH, "cache_city_tweets_ids.pickle"), "wb"))
    pickle.dump(city_to_counts, open(os.path.join(CACHE_PATH, "cache_city_tweets_counts_pro2.pickle"), "wb"))
    pickle.dump(city_to_counts_per_day, open(os.path.join(CACHE_PATH, "cache_city_tweets_counts_per_day_pro2.pickle"), "wb"))
    for i in sorted(city_to_terms):
        c = city_to_terms[i].lower()
        samples = random.sample(city_to_samples[c], min(10, len(city_to_samples[c])))
        print(c + "||" +  "||".join(samples))


def count_tweets():
    pro_count_per_day = Counter()
    anti_count_per_day = Counter()
    anti_kpop_count_per_day = Counter()
    anti_nokpop_count_per_day = Counter()
    for f in sorted(glob.iglob(BLM_TWEETS)):
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            for i,tweet in tweet_dict.items():
                contains_pro = any([x in pro_search_terms for x in  tweet['search_expressions']])
                contains_anti = any([x in anti_search_terms for x in tweet['search_expressions']])
                contains_anti = any([x in anti_search_terms for x in tweet['search_expressions']])
                contains_kpop = any([k in tweet['full_text'].lower() for k in kpop_terms])
                if contains_pro and not contains_anti:
                    pro_count_per_day[date] += 1
                if contains_anti and not contains_pro:
                    anti_count_per_day[date] += 1
                if contains_anti and contains_kpop:
                    anti_kpop_count_per_day[date] += 1
                if contains_anti and not contains_kpop and not contains_pro:
                    anti_nokpop_count_per_day[date] += 1

    for d in pro_count_per_day:
        print(d.strftime("%Y/%m/%d"), pro_count_per_day[d], anti_count_per_day[d], anti_kpop_count_per_day[d], anti_nokpop_count_per_day[d])

def drop_neverforget():
    count = 0
    old_path = "" # TO FILL IN
    new_path = "" # TO FILL IN
    for f in sorted(glob.iglob(old_path)):
        filename = os.path.basename(f)
        print(filename)
        date = get_date_from_filepath(f)
        with open(f, "rb") as fp:
            tweet_dict = pickle.load(fp)
            new_dict = {}
            for i,tweet in tweet_dict.items():
                if len(tweet['search_expressions']) == 1 and tweet['search_expressions'][0] == '#neverforget1984':
                    count += 1
                else:
                    new_dict[i] = tweet
            new_file = os.path.join(new_path, filename)
            pickle.dump(new_dict, open(new_file, 'wb'))


def main():
    # get_sample_tweets_per_city()
    drop_neverforget()

if __name__ == "__main__":
    main()
