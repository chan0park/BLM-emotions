import pickle, gzip
from collections import Counter
from process_aecl import load_relevant_aecl_data, get_events_per_city
from process_safegraph_data import load_state_names
import pandas
import random
import config
import os

# This function prints out user location strings that have U.S>
# cities in them, where the possible citites are drawn from 
# the AECL data
# This was used to generate the data to annotated for annotated_us_twitter_cities.csv
def print_possible_cities():
    user_to_location  = pickle.load(gzip.open(config.user_to_location_str))
    state_to_code = load_state_names()
    states_and_codes = list(state_to_code.keys()) + list(state_to_code.values())

    aecl_data = load_relevant_aecl_data()
    aecl_cities = get_events_per_city(aecl_data).keys()

    missing = 0
    usa_counter = Counter()
    maybe_usa = Counter()
    for u,l in user_to_location.items():
        if user_to_location[u] == "":
            missing += 1
        if any([s in l for s in states_and_codes]):
            usa_counter[l] += 1
        # elif any([c in l for c in aecl_cities]):
        #     maybe_usa[l] += 1
    pickle.dump(maybe_usa, open(os.path.join(config.CACHE_PATH, "maybe_us.pkl"), "wb"))
    maybe_usa = pickle.load(open(os.path.join(config.CACHE_PATH,"maybe_us.pkl"), "rb"))
    print("Missing", missing, missing / len(user_to_location))
    print("US states", len(usa_counter))
    print(usa_counter.most_common(200))
    fp = open("us_twitter_cities.csv", "w")
    for city,count in maybe_usa.items():
        if "England" in city or city.endswith(", UK") or "Canada" in city or "Ireland" in city or "India" in city or\
            "Panama" in city or "Mexico" in city or "Ontario" in city or "France" in city:
            continue
        if count > 100:
            fp.write("%s|%s\n" % (city, count))

def drop_nyc(c):
    return c.replace("-Bronx", "").replace("-Queens", "").replace("-Manhattan", "").replace("-Brooklyn", "").replace("-Staten Island", "").replace("New York City", "New York")

def cache_user_to_city():
    user_to_state = pickle.load(open(os.path.join(config.CACHE_PATH, "user_to_state.pkl"), "rb"))
    user_to_location  = pickle.load(gzip.open(config.user_to_location_str))

    user_str_to_state_df = pandas.read_csv(os.path.join(config.EXTRA_DATA_DIR, "annotated_us_twitter_cities.csv"))
    str_to_city = {row["Twitter String"]:row["city"] for i,row in user_str_to_state_df.iterrows() if not pandas.isnull(row["city"])}
    for s,v in str_to_city.items():
        if v in ["Queens", "Brooklyn", "Manhattan"]:
            str_to_city[s] = "New York"

    aecl_data = load_relevant_aecl_data()
    aecl_data["LOCATION"] = aecl_data["LOCATION"].apply(drop_nyc)  # We collapse NYC into one
    # nyc_set = set()
    # for l in aecl_data["LOCATION"]:
    #     if "New York" in l:
    #         nyc_set.add(l)
    # print(nyc_set)
    # return

    user_to_city = {}
    for u,state in user_to_state.items():
        user_location_str = user_to_location[u]
        if user_location_str in str_to_city:
            user_to_city[u] = str_to_city[user_location_str]
            continue

        cities = aecl_data[aecl_data["ADMIN1"] == state]
        cities = cities["LOCATION"]

        if len(cities) == 0:
            print("No data for", state)

        for c in cities:
            if c in user_to_location[u] or c.lower() in user_to_location[u]:
                user_to_city[u] = c

    print(len(user_to_city))
    pickle.dump(user_to_city, open(os.path.join(config.CACHE_PATH, "user_to_city.pkl"), "wb"))

    print("############################# Check cities ###############################")
    for u in random.sample(list(user_to_city.keys()), 20):
        print(u, user_to_city[u], user_to_location[u])

    print("############################# Check states ###############################")
    for u in random.sample(list(user_to_state.keys()), 20):
        print(u, user_to_state[u], user_to_location[u])


def cache_user_to_state():
    user_to_location  = pickle.load(gzip.open(config.user_to_location_str))
    state_to_code = load_state_names()
    code_to_state = {c:s for s,c in state_to_code.items()}
    user_str_to_state_df = pandas.read_csv("annotated_us_twitter_cities.csv")
    str_to_state = {row["Twitter String"]:row["state"] for i,row in user_str_to_state_df.iterrows() if not pandas.isnull(row["state"])}

    user_to_state = {}
    user_to_none = set()
    user_to_USA = set()
    skipped = Counter()
    multiple = {}
    for u,l in user_to_location.items():
        l = l.strip()
        possible_states = set()
        if user_to_location[u] == "":
            user_to_none.add(u)
            continue
        if l == 'United States' or l == 'USA':
            user_to_USA.add(u)
            continue

        if l in str_to_state:
            user_to_state[u] = str_to_state[l]
            continue

        for c,s in code_to_state.items():
            if l.endswith(", " + c) or l == c:
                possible_states.add(s)
        if len(possible_states) == 1:
            user_to_state[u] = possible_states.pop()
            continue

        for s,c in state_to_code.items():
            if s in l or s.lower() in l:
                if not ((s == "Virginia" and "West Virginia" in l) or \
                (s == "Nevada" and "Nevada City" in l) or \
                (s == "Delaware" and "Delaware County" in l) or \
                (s == "Wyoming" and ", PA" in l) or \
                (s == "Kansas" and "Kansas City" in l) or\
                (s == "Kansas" and ("Arkansas" in l or "arkansas" in l))):
                    possible_states.add(s)
        if len(possible_states) == 1:
            user_to_state[u] = possible_states.pop()
        elif len(possible_states) > 1:
            multiple[u] = possible_states
        else:
            skipped[l] += 1
    print("Multiple states found", len(multiple))
    for i,u in enumerate(multiple):
        print(user_to_location[u], multiple[u])
        if i > 50:
            break
    print("Found states for", len(user_to_state), len(user_to_state)/len(user_to_location))
    # print(skipped.most_common(200))
    print("None", len(user_to_none))
    print("USA", len(user_to_USA))
    pickle.dump(user_to_state, open(os.path.join(config.CACHE_PATH, "user_to_state.pkl"), "wb"))
    pickle.dump(user_to_none, open(os.path.join(config.CACHE_PATH, "user_to_none.pkl"), "wb"))
    pickle.dump(user_to_USA, open(os.path.join(config.CACHE_PATH, "user_to_USA.pkl"), "wb"))



if __name__ == "__main__":
    # Print some of the most common cities that we then annotated maunally
    # print_possible_cities()

    # First we cache the users to states.
    # cache_user_to_state()


    # Then we try to infer cities for the users that we got states for
    cache_user_to_city()
