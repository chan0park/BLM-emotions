import math
from collections import defaultdict

def write_log_odds(counts1, counts2, prior, outfile_name = None):
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1  = sum(counts1.values())
    n2  = sum(counts2.values())
    nprior = sum(prior.values())


    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if outfile_name:
      outfile = open(outfile_name, 'w')
      for word in sorted(delta, key=delta.get):
        outfile.write(word)
        outfile.write(" %.3f\n" % delta[word])

      outfile.close()
    else:
      return delta

def print_polar_words(delta, num_print):
    sorted_delta = sorted(delta.items(), key=lambda x: x[1])
    for s in sorted_delta[:num_print]:
        print(s[0], s[1])
    print("############################################################")
    sorted_delta = sorted(delta.items(), key=lambda x: x[1], reverse=True)
    for s in sorted_delta[:num_print]:
        print(s[0], s[1])
