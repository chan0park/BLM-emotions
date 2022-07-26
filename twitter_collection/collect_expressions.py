#!/usr/bin/env python3


'''
Code to search Twitter for tweets based on a list of expressions. The
    recovered tweets are stored in a file per expression in JSON format with the
    filename pattern <id>.json where <id> is the expression id indicated by the
    file expression_ids.txt .
'''


import argparse
import logging
import pprint
import sys
import traceback
import os
import twitter
import time
import json
import shutil
import gzip


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--expressions_filename', '-f',
                        required=True,
                        help='File name with the expressions to be collected.')
    parser.add_argument('--destination_dir', '-e',
                        required=True,
                        help='Directory name to be created where the collected data will be stored.')
    parser.add_argument('--credentials_filename', '-c',
                        required=True,
                        help='Filename (in JSON format) with the Twitter credentials.')
    parser.add_argument('--language', '-l',
                        default='en',
                        help='Language to filter the tweets. Default = en.')
    parser.add_argument('--max_results_per_expression', '-m',
                        type=int,
                        default=0,
                        help='Maximum number of results in a search for a expression. Default = 0 (no limit).')
    parser.add_argument('--since_id', '-i',
                        type=int,
                        default=0,
                        help='Minimum tweet ID to recover. Default = 0 (no limit).')
    parser.add_argument('--until_date', '-u',
                        default='',
                        help='Maximum date to recover. Format yyyy-mm-dd. Default = today.')
    parser.add_argument('--stop_on_error', '-s',
                        action='store_true',
                        default=False,
                        help='Stop the collecting if an HTTP error occurs. Default = no stop.')
    parser.add_argument('--debug', '-d',
                        type=int,
                        choices = [0, 1, 2],
                        nargs='?',
                        const=1,
                        default=0,
                        help='Print debug information. 0 = no debug (default); 1 = normal debug; 2 = deeper debug (HTTP debug).')
    return parser.parse_args()


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    logging.info('Starting collecting Twitter data with the following parameters:\n{}'.format(pprint.pformat(vars(args))))

    logging.info('Creating destination directory ...')
    if os.path.exists(args.destination_dir):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.mkdir(args.destination_dir)

    logging.info('Reading credentials data ...')
    with open(args.credentials_filename, mode= 'rt', encoding='ascii') as fd:
        credentials = json.load(fd)

    logging.info('Reading expressions file name ...')
    exprs = []
    expr_ids = {}
    with open(args.expressions_filename, encoding='utf-8') as fd:
        for line in fd:
            expr = line.strip().lower()     # Twitter makes no case disctintion in its search API
            if expr not in expr_ids:
                exprs.append(expr)
                expr_ids[expr] = len(expr_ids)
    with open(os.sep.join([args.destination_dir, 'expression_ids.txt']), mode='xt', encoding='utf-8') as fd:
        for expr in sorted(expr_ids.keys()):
            fd.write('{} - \'{}\'\n'.format(expr, expr_ids[expr]))
    with open(os.sep.join([args.destination_dir, 'expression_ids.json']), mode='xt', encoding='utf-8') as fd:
        json.dump(expr_ids, fd, sort_keys=True)

    logging.info('Connecting to Twitter ...')
    twitter_conn = twitter.TwitterReader(credentials['app_name'],
                                         credentials['consumer_key'],
                                         credentials['consumer_secret'],
                                         debug_connection = (args.debug == 2),
                                        )
    twitter_conn.connect()

    logging.info('Retrieving tweets ...')
    for expr in exprs:
        retry = True
        while retry:
            logging.debug(''.join(['\tSearching tweets by expression \'', expr, '\' , id = ', str(expr_ids[expr]), '...']))
            filename = os.path.join(args.destination_dir, str(expr_ids[expr]) + '.json.gz')
            if os.path.exists(filename):
                filename_backup = filename + '.backup'
                logging.warning('File {} exists (maybe due to a previous error). Backing up to {} ...'.format(filename, filename_backup))
                shutil.copy2(filename, filename_backup)
            with gzip.open(filename, mode='wt', encoding='ascii') as fd:
                try:
                    tweets = twitter_conn.search_expression(expr,
                                                            language=args.language,
                                                            max_results=args.max_results_per_expression,
                                                            retweets=True,
                                                            since_id=args.since_id,
                                                            until=args.until_date,
                                                            fd=fd,
                                                           )
                    retry = False
                except twitter.TwitterServerErrorException as tsee:
                    retry_sleep_sec = 60
                    logging.warning(''.join(['\t', str(tsee), ' Sleeping for ', str(retry_sleep_sec), ' seconds and retrying ...']))
                    time.sleep(retry_sleep_sec)
                    continue
                except Exception as e:
                    logging.error(''.join(['Error trying to search tweets by expression ', expr, ' . Error: ', str(e), ' Aborting the search for the expression \'', expr , '\' ...']))
                    traceback.print_exc()
                    if args.stop_on_error:
                        logging.error('Exiting on error ...')
                        twitter_conn.cleanup()
                        sys.exit(1)
                    twitter_conn.reconnect()

    twitter_conn.cleanup()
    logging.info('Finished.')
