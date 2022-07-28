import pandas
import os
import json
import glob
import numpy as np
import datetime
import numpy as np
from collections import Counter
from config import EXTRA_DATA_DIR

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

# access_key = B6UB6AT92CF6CV6TX4AP
# secret_access_key = eeFttP7seF0gyyIDZ5P2dG67EtprPyX7rYV2NrDL

def load_state_names(initials_as_keys = False):
    lines = open(os.path.join(EXTRA_DATA_DIR, "state_names.txt")).readlines()
    state_to_code = {}
    for l in lines:
        parts = l.split()
        state = " ".join(parts[:-1])
        code = parts[-1]
        state_to_code[state] = code
    if initials_as_keys:
        swap = {c:s for s,c in state_to_code.items()}
        swap["DC"] = "District of Columbia"
        swap["GU"] = "Guam"
        return swap

    return state_to_code

# Source of city population is U.S. Census
# https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates/2020-evaluation-estimates/2010s-cities-and-towns-total.html
def load_city_population():
    df = pandas.read_csv(os.path.join(EXTRA_DATA_DIR, "SUB-EST2020_ALL.csv"), encoding='latin-1')
    city_state_to_pop = {}
    for i,row in df.iterrows():
        city_state_to_pop[row.NAME + "_" + row.STNAME] = row.POPESTIMATE2020
    return city_state_to_pop

def process_month(data_glob, days_per_month, city_name):
    counts = np.zeros((days_per_month))
    for f in glob.iglob(data_glob):
        df = pandas.read_csv(f)
        df = df[df["city"] == city_name]
        for i,row in df.iterrows():
            visits_by_day = json.loads(row['visits_by_day'])
            counts += np.array(visits_by_day)
    return counts

def process_monthly_patterns_by_city(city_name, make_plot = False):
    monthly_patterns_may = "/usr1/home/anjalief/blm_2020/safegraph_data/monthly_patterns_May/patterns*.csv*"
    monthly_patterns_june = "/usr1/home/anjalief/blm_2020/safegraph_data/monthly_patterns_June/patterns*.csv*"
    may_counts = process_month(monthly_patterns_may, 31, city_name)
    june_counts = process_month(monthly_patterns_june, 30, city_name)

    all_counts = np.concatenate((may_counts, june_counts))

    # treat the first 21 days in May as normal data
    normal_data = (all_counts[0:7] + all_counts[7:14] + all_counts[14:21]) / 3
    series_counts = []

    curr_date = datetime.datetime(2020, 5, 1) + datetime.timedelta(days=21)
    date_to_percent_change = {}
    date_to_raw_count = {}
    date_to_subtracted = {}
    for i in range(21, len(all_counts)):
        idx = i % 7
        percent_change = (all_counts[i] / normal_data[idx]) - 1
        series_counts.append(percent_change)
        date_to_percent_change[curr_date] = percent_change
        date_to_raw_count[curr_date] = all_counts[i]
        date_to_subtracted[curr_date] = all_counts[i] - normal_data[idx]
        curr_date += datetime.timedelta(days=1)
    print(curr_date)

    # pyplot.plot(all_counts)
    # pyplot.savefig(city_name + '.png')
    if make_plot:
        pyplot.plot(series_counts)
        pyplot.savefig(city_name + '_normalized.png')
    return series_counts, date_to_percent_change, date_to_raw_count, date_to_subtracted


def load_census_blocks(use_full_states = True, verbose = False):
    df = pandas.read_csv("safegraph_data/safegraph_open_census_data/metadata/cbg_fips_codes.csv", dtype={'state_fips': str, 'county_fips': str})
    blocks_df = pandas.read_csv("safegraph_data/safegraph_open_census_data/data/cbg_b00.csv", dtype={'census_block_group': str})

    if use_full_states:
        state_to_initials = load_state_names()
        initials_to_state = {c:s for s,c in state_to_initials.items()}
        def state_to_full(s):
            if s in initials_to_state:
                return initials_to_state[s]
            return s
        df["state"] = df["state"].apply(state_to_full)

    state_to_county_count = Counter(df['state'])

    code_to_state = {}
    code_to_county = {}
    for i,row in df.iterrows():
        state = str(row["state_fips"])
        county = str(row["county_fips"])
        code_to_state[state] = row["state"]
        code_to_county[state+county] = (row["state"], row["county"].replace(" County", ""))

    county_to_block_count = Counter()
    state_to_block_count = Counter()
    county_to_population = Counter()
    state_to_population = Counter()

    # note that block groups are collections of blocks
    # blocks are an even lower-level geographic divide that census
    # does not report on
    for i,row in blocks_df.iterrows():
        group = str(row["census_block_group"])
        state_code = group[0:2]
        county_code = group[0:5]

        state = code_to_state[state_code]
        state_to_block_count[state] += 1
        if not np.isnan(row["B00001e1"]):
            state_to_population[state] += int(row["B00001e1"])

        if not county_code in code_to_county:
            if verbose:
                print("Missing census block county", county_code)
            continue

        county = code_to_county[county_code]
        county_to_block_count[county] += 1
        if not np.isnan(row["B00001e1"]):
            county_to_population[county] += int(row["B00001e1"])


    # for s,i in state_to_block_count.items():
    #     print(s, i, state_to_county_count[s], state_to_population[s])

    return state_to_county_count, county_to_block_count, state_to_block_count, state_to_population, county_to_population
    print(len(county_to_block_count))


def data_tests():
    neighborhood_data = "./safegraph_data/Neighborhood/neighborhood_patterns"
    # df = pandas.read_csv(neighborhood_data)

    daily_patterns_dir = "/usr1/home/anjalief/blm_2020/safegraph_data/daily_patterns"
    # df = pandas.read_csv(os.path.join(daily_patterns_dir, "2020-06-01-social-distancing.csv.gz"))

    monthly_patterns = "/usr1/home/anjalief/blm_2020/safegraph_data/monthly_patterns_May/patterns-part1.csv"
    df = pandas.read_csv(monthly_patterns)
    print(len(df))
    print(df.columns)
    for i,row in df.iterrows():
        print(row)
        print(len(json.loads(row['visits_by_day'])))
        # print(len(json.loads(row['popularity_by_hour_monday'])))
        break

def main():
    counts, date_to_percent_change = process_monthly_patterns_by_city("Philadelphia")
    for x in sorted(date_to_percent_change):
        print(x, date_to_percent_change[x])


if __name__ ==  "__main__":
    # main()
    load_census_blocks()