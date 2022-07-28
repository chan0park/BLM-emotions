import pandas
from collections import Counter, defaultdict
import datetime
from stat_utils import *
matplotlib.use('Agg')
from matplotlib import pyplot
from process_safegraph_data import load_census_blocks, load_state_names, load_city_population
import numpy as np
from config import EXTRA_DATA_DIR
import os

SIZE_KEY = "size_high"
# SIZE_KEY = "size_low"

CCC_DATA = os.path.join(EXTRA_DATA_DIR, "ccc_compiled.csv")
KEEP_ISSUES = [' racism', 'policing', ' policing', 'racism', ' criminal justice', 'criminal justice']

def str_to_date(d):
    return datetime.datetime.strptime(d, '%Y-%m-%d')

def get_city_to_protest_count():
    ccc_data = load_relevant_ccc_data()
    ccc_data = ccc_data.dropna(subset=['state', "locality"])
    city_counter = Counter()
    for _,row in ccc_data.iterrows():
        city_counter[row["locality"] + "_" + row["state"]] += 1
    return dict(city_counter)

# Note that the ordering of states in this output doesn't really make sense, normalizing
# by population at a state level is too strong (it just ranks the least populous states)
# as having the most number of protests
def get_state_to_protest_size_normalized():
    ccc_data = load_relevant_ccc_data()
    ccc_data = ccc_data.dropna(subset=['state'])
    print(len(ccc_data))
    nan_filtered = ccc_data.dropna(subset=[SIZE_KEY])
    state_to_mean = defaultdict(list)
    for i,row in nan_filtered.iterrows():
        state_to_mean[row['state']].append(row[SIZE_KEY])
    state_to_mean = {s:np.mean(l) for s,l in state_to_mean.items()}
    print(state_to_mean)
    ccc_data[SIZE_KEY] = ccc_data.apply(lambda x: state_to_mean[row["state"]] if pandas.isnull(row[SIZE_KEY]) else row[SIZE_KEY], axis=1)
    ccc_data = ccc_data[["state", SIZE_KEY]]
    ccc_data.groupby(['state']).sum()
    _, _, _, state_to_population, _ = load_census_blocks()
    state_counter = {row['state']:row[SIZE_KEY]/state_to_population[row['state']] for _,row in ccc_data.iterrows() if row['state'] in state_to_population}
    return dict(state_counter)

def get_city_to_protest_size_normalized():
    ccc_data = load_relevant_ccc_data()
    ccc_data = ccc_data.dropna(subset=['state', "locality"])
    nan_filtered = ccc_data.dropna(subset=[SIZE_KEY])

    city_to_mean = defaultdict(list)
    for i,row in nan_filtered.iterrows():
        city_to_mean[row["locality"] + "_" + row["state"]].append(row[SIZE_KEY])
    city_to_mean = {s:np.mean(l) for s,l in city_to_mean.items()}

    city_to_sum = defaultdict(list)
    num_null = 0
    for _,row in ccc_data.iterrows():
        key = row["locality"] + "_" + row["state"]
        
        if pandas.isnull(row[SIZE_KEY]):
            num_null += 1
            if key in city_to_mean:
                city_to_sum[key].append(city_to_mean[key])
        else:
            city_to_sum[key].append(row[SIZE_KEY])
    print("Number of null size", num_null, len(ccc_data), num_null/len(ccc_data))
    city_to_sum = {s:np.sum(l) for s,l in city_to_sum.items()}

    city_to_population = load_city_population()
    city_to_population = {k.replace(" city", "").replace(" town", ""):n for k,n in city_to_population.items() if n > 50000}
    city_to_norm = {c:n/city_to_population[c] for c,n in city_to_sum.items() if c in city_to_population}

    return city_to_norm


def get_state_to_protest_count(normalize_by_county = False):
    # 'District of Columbia'
    ccc_data = load_relevant_ccc_data()
    ccc_data = ccc_data.dropna(subset=['state'])
    state_counter = Counter(ccc_data["state"])
    if normalize_by_county:
        state_to_county_count, _, _, _, _ = load_census_blocks()
        state_counter = {s:c/state_to_county_count[s] for s,c in state_counter.items() if s in state_to_county_count}
    return dict(state_counter)

# This is volume of tweets from our spreadsheet
def load_tweet_volume():
    lines = open("./external_data/tweet_counts.txt").readlines()
    date_to_count = {}
    for l in lines:
        parts = l.split()
        date = datetime.datetime.strptime(parts[0], "%Y/%m/%d")
        count = int(parts[1])
        date_to_count[date] = count
    return date_to_count

def is_blm(row):
    issues = row["issues"].split()
    return any([i in KEEP_ISSUES for i in issues])

def plot_series(d, filename, use_daterange = False):
    data = []
    if use_daterange:
        date = datetime.datetime(year=2020, month=5, day=24)
        while date <= datetime.datetime(year=2020, month=6, day=30):
            if not date in d:
                data.append(0)
            else:
                data.append(d[date])
            date += datetime.timedelta(days=1)
    else:
        for i in sorted(d):
            data.append(d[i])
    pyplot.plot(data)
    pyplot.savefig(filename)

def load_relevant_ccc_data(verbose=False, event_type = None):
    df = pandas.read_csv(CCC_DATA, encoding='latin-1')
    if verbose:
        print("Number of unfiltered events", len(df))
        # print(df.columns)
    df = df.dropna(subset=['date', 'issues'])
    print("Number of unfiltered events", len(df))

    # Cut to relevant dates
    firstdate = str_to_date('2020-05-24')
    lastdate = str_to_date('2020-06-30')
    df['date'] = df['date'].apply(str_to_date)
    df = df[df['date'] >= firstdate]
    df = df[df['date'] <= lastdate]

    if verbose:
        print("Number of events in our date range", len(df))
    df = df[df.apply(is_blm, axis=1)]
    if verbose:
        print("Number of BLM events in our date range", len(df))

    code_to_state = load_state_names(initials_as_keys=True)
    df["state"] = df["state"].apply(lambda x: code_to_state.get(x, x))
    return df


def write_events_per_day():
    ccc_data = load_relevant_ccc_data()
    print(ccc_data.columns)
    ccc_data["date_str"] = ccc_data["date"].apply(lambda x: "%s/%s" %(x.month, x.day))
    counts = Counter(ccc_data["date_str"])
    df = pandas.DataFrame({"date": counts.keys(), "Protest Count" : counts.values()})
    df.to_csv("data/ccc_protest_counts.csv", header=True, columns=["date","Protest Count"], index=False)

def main():
    ss = get_state_to_protest_count(normalize_by_county = True)
    # print(cc)
    # ss = get_state_to_protest_size_normalized()
    for s in sorted(ss.items(), key=lambda x: x[1]):
        print(s)

    # city_to_norm = get_city_to_protest_size_normalized()
    # sorted_cities = sorted(city_to_norm.items(), key=lambda x: x[1])
    # print(len(city_to_norm))
    # for s in sorted_cities[:20]:
    #     print(s)
    # print('...')
    # for s in sorted_cities[-20:]:
    #     print(s)
    # # write_events_per_day()

if __name__ == "__main__":
    main()
    # print_state_to_count()
