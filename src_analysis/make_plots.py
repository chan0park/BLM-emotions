import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas
import matplotlib.dates as mdates
import numpy as np
import argparse

plt.rcParams.update({'font.size': 15})

emotion_to_color = {
    # "anger_disgust": ("#DDCC77", "|", "Anger/Disgust"),
    "anger": ("#332288", "s", "Anger"),
    "disgust": ("#882255", "o", "Disgust"),
    "joy": ("#88CCEE", "P", "Positivity"),
    "surprise": ("#CC6677", "x", "Surprise"),
    "sadness": ("#44AA99", "d", "Sadness"),
    'fear': ("#DDCC77","^", "Fear"),

}

sentiment_to_color = {
    "negative": ("#332288", "s", "Negative"),
    "positive": ("#88CCEE", "P", "Positive"),
}

tweet_to_color = {
    "# tweets" : ("#882255", "x"),
    "# retweets" : ("#CC6677", "|")
}

protest_color = "#88CCEE"
protest_marker = "P"

ccc_protest_color = "#44AA99"

user_to_color = {
    "# users" : ("#332288", "s"),
    "# new users" : ("#88CCEE", "o")
}

def emotions_by_date(df, save_name, number, label_to_color, normalize, ylabel):
    df["date"] = pandas.to_datetime(df[df.columns[1]])

    fig = plt.figure(number)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', labelrotation=50)

    for e, params in label_to_color.items():
        if not e in df.columns:
            continue

        color, marker, l = params
        if normalize:
            vals = df.apply(lambda row: row[e] / row["date_counts"] * 100, axis=1)
        else:
            vals = df[e]
        plt.plot(df["date"], vals, label=l, color=color, marker=marker)

    # leg = plt.legend(loc='best', ncol=2)
    plt.legend(bbox_to_anchor=(1.1, 1.35), ncol=3)
    # leg.get_frame().set_alpha(0.5)

    plt.ylabel(ylabel)
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_name)

def tweet_counts(df, save_name, number):
    df["date"] = [x + "/2020" for x in df[df.columns[0]]]
    df["date"] = [x.to_pydatetime() for x in pandas.to_datetime(df["date"])]
    dates = [x.to_pydatetime() for x in df["date"]]

    fig = plt.figure(number)
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=50)

    for t, color_marker in tweet_to_color.items():
        color, marker = color_marker
        vals = df[t].astype(float) # apply(lambda x: int(x.replace("%", "")))
        ax1.plot(dates, vals, label=t, color=color) # marker=marker)

    leg = plt.legend(loc='upper right', ncol=1)


    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.set_ylabel("Millions of tweets")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    for u, color_marker in user_to_color.items():
        color, marker = color_marker
        vals = df[u].astype(float) # apply(lambda x: int(x.replace("%", "")))
        ax2.plot(dates, vals, label=u, color=color) #marker=marker)# secondary_y=True)
    ax2.set_ylabel("Millions of users")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    leg = plt.legend(loc='center right', ncol=1)
    plt.legend(bbox_to_anchor=(1.0, 0.75))
    #leg.get_frame().set_alpha(0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_name)


def protest_volume(aecl_df, ccc_df, save_name, number):
    def process_df(df):
        df["date"] = [x + "/2020" for x in df[df.columns[0]]]
        df["date"] = [x.to_pydatetime() for x in pandas.to_datetime(df["date"])]
        df = df.sort_values(by=['date'])
        dates = [x.to_pydatetime() for x in df["date"]]
        return df, dates

    df, dates = process_df(aecl_df)

    fig = plt.figure(number)
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=50)

    vals = df["# tweets"].astype(float) # apply(lambda x: int(x.replace("%", "")))
    color, marker = tweet_to_color["# tweets"]
    ax1.plot(dates, vals, label="# tweets", color=color) # marker=marker)
    leg = plt.legend(loc='upper right', ncol=1)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.set_ylabel("Millions of tweets")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    vals = df["Protest Count"].astype(float) # apply(lambda x: int(x.replace("%", "")))
    ax2.plot(dates, vals, label="ACLED # Protests", color=protest_color) # marker=protest_marker)

    ccc_df, ccc_dates = process_df(ccc_df)
    ax2.plot(ccc_dates, ccc_df["Protest Count"].astype(float), label="CCC # Protests", color=ccc_protest_color) # marker=protest_marker)

    leg = plt.legend(loc='upper right', ncol=1)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.set_ylabel("Number of U.S. Protests")

    plt.legend(bbox_to_anchor=(.37, 0.87))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_name)


def by_keyword(df, save_name, number, label_to_color, ylabel):
    fig = plt.figure(number)
    ax = fig.add_subplot(111)

    labels = ["all", "pro-BLM", "anti-BLM", "police", "protests"]
    barWidth = 0.15
    r1 = np.arange(len(labels))
    # Create bars
    emotion_to_row = {row[1]:row for i,row in df.iterrows()}
    for emotion,specs in label_to_color.items():
        if not emotion in emotion_to_row:
            continue
        row = emotion_to_row[emotion]
        color, _, label = specs
        vals = [row[c] for c in labels]
        ax.bar(r1, vals, width = barWidth, color = color, edgecolor = 'black', label=label)

        # Set position for next series
        r1 = [x + barWidth for x in r1]

    mid_point = barWidth * (len(labels) / 2.0) - 0.5 * barWidth
    plt.xticks([r + mid_point for r in range(len(labels))], labels)
 
    # leg = plt.legend(loc='upper center', ncol=3)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.0, 1.25), ncol=3)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_name)

def emotions2014(data, save_name, number):
    fig = plt.figure(number)
    ax = fig.add_subplot(111)

    labels = ["2012-2015", "2020"]
    barWidth = 0.15
    r1 = np.arange(len(labels))

    # Create bars
    for emotion,vals in data.items():
        if not emotion in emotion_to_color:
            continue
        color, _, l = emotion_to_color[emotion]
        ax.bar(r1, vals, width = barWidth, color = color, edgecolor = 'black', label=l)

        # Set position for next series
        r1 = [x + barWidth for x in r1]

    mid_point = barWidth * (len(labels) / 2.0) + barWidth
    plt.xticks([r + mid_point for r in range(len(labels))], labels)
 
    # leg = plt.legend(loc='upper center', ncol=3)
    plt.ylabel("% of tweets containing each emotion")
    plt.legend(bbox_to_anchor=(1.0, 1.25), ncol=3)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", choices=['ekman', 'vader', "sentiment140", "pcc", "other"], help="typically will be ekman to replicate paper figures")
    parser.add_argument("--file_type", choices=['png', 'pdf'], help="specify if output should be saved as pdf or png")
    args = parser.parse_args()

    if args.key in ["ekman", "pcc"]:
        label_to_color = emotion_to_color
    else:
        label_to_color = sentiment_to_color

    if args.key == "pcc":
        ylabel = "Avg. emotion probability estimate"
        normalize = False
    else:
        ylabel = "% of tweets containing each emotion"
        normalize = True

    if args.key != "other":
        df = pandas.read_csv("data/emotions_over_time.%s.csv" % args.key)
        emotions_by_date(df, "plots/emotions_over_time.%s.%s" % (args.key, args.file_type), 1, label_to_color, normalize, ylabel)

        df = pandas.read_csv("data/pro_emotions_over_time.%s.csv" % args.key)
        emotions_by_date(df, "plots/pro_emotions_over_time.%s.%s" % (args.key, args.file_type), 2, label_to_color, normalize, ylabel)

        df = pandas.read_csv("data/anti_emotions_over_time.%s.csv" % args.key)
        emotions_by_date(df, "plots/anti_emotions_over_time.%s.%s" % (args.key, args.file_type), 6, label_to_color, normalize, ylabel)

        df = pandas.read_csv("data/keyword_split.%s.csv" % args.key)
        by_keyword(df, "plots/emotions_by_topic.%s.%s" % (args.key, args.file_type), 5, label_to_color, ylabel)

    else:
        # df = pandas.read_csv("data/tweet_counts.csv")
        # tweet_counts(df, "plots/data_distribution.pdf", 3)

        df = pandas.read_csv("data/protest_counts.csv")
        ccc_df = pandas.read_csv("data/ccc_protest_counts.csv")
        protest_volume(df, ccc_df, "plots/protest_tweet_count.%s" % args.file_type, 4)


        # data_2014 = {
        #     "anger": [32.68, 43.15],
        #     "disgust": [6.70, 18.68],
        #     "joy": [53.84, 48.31],
        #     "surprise": [2.55, 4.39],
        #     "sadness": [4.11,	3.56]
        # }


        # data_2014 = {
        #     "anger": [13.1, 44.53],
        #     "disgust": [4.67, 27.21],
        #     "joy": [53.18, 27.037],
        #     "surprise": [2.13, 6.61],
        #     "sadness": [6.2,	4.77],
        #     "fear": [10.9,	7.45]
        # }

        # emotions2014(data_2014, "plots/2014.%s" % args.file_type, 6)

def print_counts():
    df = pandas.read_csv("data/emotions_over_time.ekman.csv")
    print("All", sum(df["date_counts"]))

    df = pandas.read_csv("data/pro_emotions_over_time.ekman.csv")
    print("Pro", sum(df["date_counts"]))

    df = pandas.read_csv("data/anti_emotions_over_time.ekman.csv")
    print("Anti", sum(df["date_counts"]), sum(df["sadness"]))


if __name__ == "__main__":
    main()
    # print_counts()
