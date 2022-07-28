import pandas
import argparse
from collections import Counter, defaultdict
import datetime
from stat_utils import *
matplotlib.use('Agg')
from matplotlib import pyplot
from process_safegraph_data import process_monthly_patterns_by_city, load_census_blocks, load_city_population
import numpy as np


AECL_DATA = "./external_data/ACLED_data.csv"

def str_to_date(d):
    return datetime.datetime.strptime(d, '%d-%B-%Y')

def get_city_to_protest_count():
    aecl_data = load_relevant_aecl_data()
    state_counter = Counter()
    for i,row in aecl_data.iterrows():
        if "New York" in row["LOCATION"]:
            city = "New York"
        else:
            city = row["LOCATION"]
        state_counter[city + "_" + row["ADMIN1"]] += 1
    return dict(state_counter)

def get_city_to_protest_size_normalized():
    aecl_data = load_relevant_aecl_data()
    aecl_data = populate_size_estimates(aecl_data)


    aecl_data["LOCATION"] = aecl_data["LOCATION"].apply(lambda x: "New York" if "New York" in x else x)

    nan_filtered = aecl_data[[type(x) != str for x in aecl_data["size"]]]

    city_to_mean = defaultdict(list)
    print(len(nan_filtered))
    for _,row in nan_filtered.iterrows():
        city_to_mean[row["LOCATION"] + "_" + row["ADMIN1"]].append(row['size'])
    city_to_mean = {s:np.mean(l) for s,l in city_to_mean.items()}

    city_to_sum = defaultdict(list)
    num_null = 0
    for _,row in aecl_data.iterrows():
        key = row["LOCATION"] + "_" + row["ADMIN1"]
        
        if type(row["size"]) == str:
            num_null += 1
            if key in city_to_mean:
                city_to_sum[key].append(city_to_mean[key])
        else:
            city_to_sum[key].append(row["size"])
    print("Number of null size", num_null, len(aecl_data), num_null/len(aecl_data))
    city_to_sum = {s:np.sum(l) for s,l in city_to_sum.items()}

    city_to_population = load_city_population()
    city_to_population = {k.replace(" city", "").replace(" town", ""):n for k,n in city_to_population.items() if n > 50000}
    city_to_norm = {c:n/city_to_population[c] for c,n in city_to_sum.items() if c in city_to_population}

    return city_to_norm

def get_state_to_protest_count(normalize_by_county = False):
    aecl_data = load_relevant_aecl_data()
    state_counter = Counter(aecl_data["ADMIN1"])
    if normalize_by_county:
        state_to_county_count, _, _, _, _ = load_census_blocks()
        state_to_county_count["District of Columbia"] = state_to_county_count["DC"]
        state_counter = {s:c/state_to_county_count[s] for s,c in state_counter.items()}
    return dict(state_counter)


def get_events_per_day(df, verbose = False):
    counts_per_day = Counter(df['EVENT_DATE'])
    if verbose:
        for d in sorted(counts_per_day):
            print("%s %s" % (d.strftime("%Y/%m/%d"), counts_per_day[d]))
    return counts_per_day

def get_events_per_city(df, verbose = False):
    return Counter(df['LOCATION'])

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
    # it's a float if it's Nan
    if type(row['ASSOC_ACTOR_1']) != float:
        if "BLM" in row['ASSOC_ACTOR_1'] or "Black Lives Matter" in row['ASSOC_ACTOR_1']:
            return True
    if type(row['ASSOC_ACTOR_2']) != float:
        if "BLM" in row['ASSOC_ACTOR_2'] or "Black Lives Matter" in row['ASSOC_ACTOR_2']:
            return True
    return False

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

def compare_aecl_and_safegraph(df, city_name, firstdate):
    city_df = df[df['LOCATION'] == city_name]
    city_counts = get_events_per_day(city_df, verbose=False)
    # plot_series(city_counts, "%s_aecl_perday.png" % city_name)

    _, date_to_percent_change, date_to_raw_count, date_to_subtracted = process_monthly_patterns_by_city(city_name)
    for d in date_to_percent_change:
        if not d in city_counts and d > firstdate:
            city_counts[d] = 0
    s1, s2 = make_series(date_to_percent_change, city_counts)
    print("Correlation between AECL and SafeGraph percent change for %s" % city_name, get_corr(s1, s2))

    s1, s2 = make_series(date_to_raw_count, city_counts)
    print("Correlation between AECL and SafeGraph raw counts for %s" % city_name, get_corr(s1, s2))

    s1, s2 = make_series(date_to_subtracted, city_counts)
    print("Correlation between AECL and SafeGraph subtracted for %s" % city_name, get_corr(s1, s2))


# Event types: Violent demonstration, Peaceful protest, Excessive force against protesters
def load_relevant_aecl_data(verbose=False, event_type = None):
    df = pandas.read_csv(AECL_DATA)
    if verbose:
        print("Number of unfiltered events", len(df))
        print(df.columns)

    # Cut to relevant dates
    firstdate = str_to_date('24-May-2020')
    lastdate = str_to_date('30-June-2020')
    df['EVENT_DATE'] = df['EVENT_DATE'].apply(str_to_date)
    df = df[df['EVENT_DATE'] >= firstdate]
    df = df[df['EVENT_DATE'] <= lastdate]


    if verbose:
        print("Number of events in our date range", len(df))
    df = df[df.apply(is_blm, axis=1)]
    if verbose:
        print("Number of BLM events in our date range", len(df))
    if event_type is not None:
        df = df[df['SUB_EVENT_TYPE'] == event_type]
        print("Number of %s" %event_type, len(df))
    return df

def print_state_to_count():
    df = load_relevant_aecl_data()
    c = Counter(df["ADMIN1"])
    for state,c in c.items():
        print(state + ",", c)


def populate_size_estimates(df):
    str_to_size = {'hundreds': 500,
    'dozens' : 60,
    'thousands': 5000,
    'several hundred': 300,
    'a few hundred': 300,
    'several dozen': 36,
    'about a dozen': 12
    }

    def get_size(notes):
        # it looks like "...amid the coronavirus pandemic. [size=almost 1 dozen]"
        text_str = notes.split("size=")[-1][:-1]
        if "]":
            text_str = text_str.split("]")[0]
        parts = text_str.split()
        parts = [p.split("-") for p in parts]
        parts = [item.replace(",", "") for sublist in parts for item in sublist]
        size_estimate = [int(s) for s in parts if s.isdigit()]
        if len(size_estimate) > 1:
            return max(size_estimate)
        elif len(size_estimate) == 1:
            return int(size_estimate[0])
        elif text_str in str_to_size:
            return int(str_to_size[text_str])
        else:
            return ""

    df["size"] = df["NOTES"].apply(get_size)
    return df


def main():
    # df = load_relevant_aecl_data()
    # events_per_day = get_events_per_day(df)
    # date_to_tweet_count = load_tweet_volume()
    # s1, s2 = make_series(events_per_day, date_to_tweet_count)
    # print("Correlation between events per day and tweet volume", get_corr(s1, s2))

    # # First column is the intercept value
    # # Last column is the p-value
    # print("Check if tweet volumne can predict protests")
    # do_granger_R(s2, s1)
    # print("Check if protests can predict tweet volume")
    # do_granger_R(s1, s2)
    # # do_curve_seasonal_adjust(events_per_day)

    # location_counts = get_events_per_city(df)
    # print(len(location_counts))
    # print("Philadelphia", location_counts["Philadelphia"])
    # popular_location = {l:c for l,c in location_counts.items() if c >= 10}
    # for p,c in popular_location.items():
    #     print(p, c)

    # compare_aecl_and_safegraph(df, "Philadelphia", firstdate)
    # compare_aecl_and_safegraph(df, "Chicago", firstdate)
    
    # city_to_norm = get_city_to_protest_size_normalized()
    city_to_norm = get_state_to_protest_count(normalize_by_county=True)
    sorted_cities = sorted(city_to_norm.items(), key=lambda x: x[1])
    print(len(city_to_norm))
    for s in sorted_cities[:20]:
        print(s)
    print('...')
    for s in sorted_cities[-20:]:
        print(s)



if __name__ == "__main__":
    main()
    # print_state_to_count()
