import pandas
import logging
import pickle
import numpy
import sklearn.metrics
import pprint



def calculate_accuracies(gt, preds):
    accuracies = []
    for dimension in range(gt.shape[1]):
        accuracies.append(sklearn.metrics.accuracy_score(gt[:,dimension], preds[:,dimension]))
    return numpy.asarray(accuracies)


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')


logging.info('Reading data ...')
annotated_data = {}
liwc_data = {}
for split, filename in [('train', 'train_liwc_outputs.csv'),
                        ('dev', 'dev_liwc_outputs.csv'),
                        ('test', 'test_liwc_outputs.csv'),
                       ]:
    annotated_liwc_data = pandas.read_csv(filename)
    annotated_data[split] = []
    liwc_data[split] = []
    for _, row in annotated_liwc_data.iterrows():
        annotated_data[split].append([row['anger'],
                                      row['joy'],
                                      row['sadness'],
                                     ])
        liwc_data[split].append([row['emo_anger'],
                                 row['emo_pos'],
                                 row['emo_sad'],
                                ])
    annotated_data[split] = numpy.asarray(annotated_data[split])
    liwc_data[split] = numpy.asarray(liwc_data[split])

liwc_data['train+dev'] = numpy.vstack((liwc_data['train'], liwc_data['dev']))
annotated_data['train+dev'] = numpy.vstack((annotated_data['train'], annotated_data['dev']))


split = 'test'
logging.info('LIWC - Scores with naive threshold (0.0) over {} data:'.format(split))
naive_classifier = numpy.vectorize(lambda x: int(x>0.0))
liwc_naive_test_preds = naive_classifier(liwc_data[split])
logging.info('\tAccuracy score: {}'.format(sklearn.metrics.accuracy_score(annotated_data[split], liwc_naive_test_preds)))
accuracies = calculate_accuracies(annotated_data[split], liwc_naive_test_preds)
logging.info('\tIndividual accuracy scores (anger, joy, sadness): {}'.format(accuracies))
logging.info('\tAccuracy average score: {}'.format(accuracies.mean()))
logging.info('\tF1 macro score: {}'.format(sklearn.metrics.f1_score(annotated_data[split], liwc_naive_test_preds, average='macro')))
logging.info('\tF1 micro score: {}'.format(sklearn.metrics.f1_score(annotated_data[split], liwc_naive_test_preds, average='micro')))
logging.info('\tF1 individual scores (anger, joy, sadness): {}'.format(pprint.pformat(sklearn.metrics.f1_score(annotated_data[split], liwc_naive_test_preds, average=None))))

logging.info('LIWC - Searching for optimal thresholds with train+dev data ...')
optimal_thresholds = []
x_train_data = liwc_data['train+dev']
y_train_data = annotated_data['train+dev']
for dimension in range(y_train_data.shape[1]):
    min_value = x_train_data.min()
    max_value = x_train_data.max()
    best_threshold = min_value
    best_f1 = 0.0
    for threshold in numpy.arange(min_value,
                                  max_value,
                                  (max_value-min_value)/10_000,
                                 ):
        classifier = numpy.vectorize(lambda x: int(x>threshold))
        preds = classifier(x_train_data[:,dimension])
        f1_score = sklearn.metrics.f1_score(y_train_data[:,dimension], preds, average='macro')
        logging.debug('\tF1 macro score: for dimension={} and threshold={}: {}'.format(dimension, threshold, f1_score))
        if best_f1 < f1_score:
            best_f1 = f1_score
            best_threshold = threshold
            logging.debug('\t\tBest threshold value found!')
    optimal_thresholds.append(best_threshold)
logging.info('Best thresholds found (anger, joy, sadness): {}'.format(pprint.pformat(optimal_thresholds)))

splits = ['train+dev', 'test']
for split in splits:
    logging.info('LIWC - Scores with best threshold over {} data:'.format(split))
    preds = []
    for dimension in range(annotated_data[split].shape[1]):
        classifier = numpy.vectorize(lambda x: int(x>optimal_thresholds[dimension]))
        preds.append(classifier(liwc_data[split][:,dimension]).reshape(-1, 1))
    preds = numpy.hstack(preds)
    logging.info('\tAccuracy score: {}'.format(sklearn.metrics.accuracy_score(annotated_data[split], preds)))
    accuracies = calculate_accuracies(annotated_data[split], preds)
    logging.info('\tIndividual accuracy scores (anger, joy, sadness): {}'.format(accuracies))
    logging.info('\tAccuracy average score: {}'.format(accuracies.mean()))
    logging.info('\tF1 macro score: {}'.format(sklearn.metrics.f1_score(annotated_data[split], preds, average='macro')))
    logging.info('\tF1 micro score: {}'.format(sklearn.metrics.f1_score(annotated_data[split], preds, average='micro')))
    logging.info('\tF1 individual scores (anger, joy, sadness): {}'.format(pprint.pformat(sklearn.metrics.f1_score(annotated_data[split], preds, average=None))))
    logging.info('\tPrecision macro score: {}'.format(sklearn.metrics.precision_score(annotated_data[split], preds, average='macro')))
    logging.info('\tPrecision micro score: {}'.format(sklearn.metrics.precision_score(annotated_data[split], preds, average='micro')))
    logging.info('\tPrecision individual scores (anger, joy, sadness): {}'.format(pprint.pformat(sklearn.metrics.precision_score(annotated_data[split], preds, average=None))))
    logging.info('\tRecall macro score: {}'.format(sklearn.metrics.recall_score(annotated_data[split], preds, average='macro')))
    logging.info('\tRecall micro score: {}'.format(sklearn.metrics.recall_score(annotated_data[split], preds, average='micro')))
    logging.info('\tRecall individual scores (anger, joy, sadness): {}'.format(pprint.pformat(sklearn.metrics.recall_score(annotated_data[split], preds, average=None))))

logging.info('Finished.')