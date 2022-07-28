import numpy
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from numpy import polyfit


def get_corr(x, y):                                                                                                                                                                  
    return pearsonr(x, y)

# Turn two dictionaries into parallel time series
def make_series(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    joint_keys = d1_keys.intersection(d2_keys)
    series1 = []
    series2 = []
    for k in sorted(joint_keys):
        series1.append(d1[k])
        series2.append(d2[k])
    return series1, series2

def average(s):
    return sum(s) / len(s)

# take % change of series
def make_percent_change(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        diff.append((dataset[i] / dataset[i - interval]) - 1)
    return diff

def do_curve_seasonal_adjust(series_dict):
    # fit polynomial: x^2*b1 + x*b2 + ... + bn
    X = []
    y = []
    for i,x in enumerate(sorted(series_dict)):
        y.append(series_dict[x])
        X.append(i)
    degree = 4
    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
	    value = coef[-1]
	    for d in range(degree):
		    value += X[i]**(degree-d) * coef[d]
	    curve.append(value)
    # plot curve over original data
    pyplot.plot(y)
    pyplot.plot(curve, color='red', linewidth=3)
    pyplot.savefig('curve_series.png')
