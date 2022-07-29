This directory contains the primary code used for analysis. It assumes the existence of data caches (raw data and emotion-labeled tweets)

Caches to data and emotions labels must be set in config.py for the code to be runnable

The file `process_emotions.py` is the primary file for generating the main results in the paper. Setting `--key ekman` and specifying which experiment to run will reproduce paper results. For example, `--filtered_over_time` will compute percentages of emotions in tweets with pro-BLM hashtags over time. This file writes outputs to the cache directory specified in config.py and the data directory

Once `process_emotions.py` has been run, `make_plots.py` can be run to graph the saved outputs

`process_user_locations.py` must be run in order to compute geographic correlations (this caches inferred user geographic locations). After the necessary caches have been created, it should be possible to run `process_emotions.py --key ekman --print_location_corrs`

`pcc_process_emotions.py` is extremely similar to `process_emotions.py` but draws from raw model output scores and can be used to recreated probablistic classify and count results in the appendix.

Other files generally contain helper functions


Note: some of our early data collection erroneuosly included tweets with the hashtag '#neverforget1984'. These tweets can be remove with the helper function in load_filtered_tweets.py
