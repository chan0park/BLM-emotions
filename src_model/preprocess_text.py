#!/usr/bin/env python3
# TODO: identify positive/negative emoticons and tag them


import argparse
import logging
import pprint
import os
import glob
import re
import string
import html
from functools import reduce, partial

import emoji
from wordsegment import load, segment


URL_REGEX = '((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)'
# URL_TAG = u'http'
URL_TAG = ''

USER_REGEX = '@[^\s]+'
# USER_TAG = u'@USER'
USER_TAG = ''

NUMBER_REGEX = '[0-9]([0-9,.]*[0-9])?'
# NUMBER_TAG = u'<NUMBER>'
NUMBER_TAG = ''


def command_line_parsing():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_directory', '-i',
                        required=True,
                        help='Directory with text to be masked.')
    parser.add_argument('--output_directory', '-o',
                        required=True,
                        help='Output directory.')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def compose(*funcs):
    """" Compose functions so that they are applied in chain. """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])


def normalize_quotes(text):
    text = re.sub(r'[\u2018\u2019]', '\'', text)    # single quotes
    text = re.sub(r'[\u201C\u201D]', '\"', text)    # double quotes
    return text


def tag(text, regex, tag):
    tokens = []
    for token in text.split():
        if re.fullmatch(regex, token):
            tokens.append(tag)
        else:
            tokens.append(token)
    return ' '.join(tokens)


def add_capital_signs(text):
    def _has_cap(token):
        return token.lower() != token and token.upper() != token

    def _all_cap(token):
        return token.lower() != token and token.upper() == token
    exceptions = ['@USER', 'URL']
    tokens = text.split()
    # tokens = ['<has_cap> ' +
    #           t if _has_cap(t) and t not in exceptions else t for t in tokens]
    # tokens = ['<all_cap> ' +
    #           t if _all_cap(t) and t not in exceptions else t for t in tokens]
    tokens = [
        t if _has_cap(t) and t not in exceptions else t for t in tokens]
    tokens = [
        t if _all_cap(t) and t not in exceptions else t for t in tokens]
    return ' '.join(tokens)


def _limit_pattern(sent, pattern, keep_num):
    if pattern in string.punctuation:
        re_pattern = re.escape(pattern)
    else:
        re_pattern = f'(({pattern})[\s]*)'
        pattern = pattern + ' '
    pattern_regex = re_pattern + '{' + str(keep_num+1) + ',}'
    return re.sub(pattern_regex, lambda match: pattern * keep_num, sent)


def limit_punctuations(sent, keep_num):
    puncs = ['!', '?', '.']
    for p in puncs:
        sent = _limit_pattern(sent, p, keep_num)
    return sent


def limit_mentions(sent, keep_num):
    return _limit_pattern(sent, '@USER', keep_num)


def replace_emojis(sent):
    """ e.g. smiling emoticon -> :smiley_face: """
    return emoji.demojize(sent)


def textify_emojis(sent):
    """ e.g. :smiley_face: -> smiley face"""
    return re.sub(':[\S]+:', lambda match: match.group().replace('_', ' ').replace('-', ' ').replace(':', ' '), sent)
    #ret = re.sub(':[\S]+:', lambda match: match.group().replace('_', ' ').replace(':', ''), sent)
    # return '<emoji> ' + ret + ' </emoji>'


def lower_hashtags(sent):
    """ e.g.  #MAGA -> #maga """
    return re.sub('#[\S]+', lambda match: match.group().lower(), sent)


def segment_hashtags(sent):
    """ e.g. #MakeAmericaGreatAgain -> make america great again"""
    return re.sub('#[\S]+', lambda match: ' '.join(segment(match.group())), sent)
    # ret = re.sub('#[\S]+', lambda match: ' '.join(segment(match.group())), sent)
    # return '<hashtag> ' + ret + ' </hashtag>'


def build_preprocess(demojize, textify_emoji, mention_limit, punc_limit, lower_hashtag,
                     segment_hashtag, add_cap_sign):
    if textify_emoji and not demojize:
        raise Exception("textify_emoji is meaningless without demojize")

    funcs = [
        html.unescape,
        normalize_quotes,
        partial(tag, regex=URL_REGEX, tag=URL_TAG),
        partial(tag, regex=USER_REGEX, tag=USER_TAG),
        partial(tag, regex=NUMBER_REGEX, tag=NUMBER_TAG),
    ]

    if demojize:
        funcs.append(replace_emojis)
    if textify_emoji:
        funcs.append(textify_emojis)
    if mention_limit > 0:
        funcs.append(partial(limit_mentions, keep_num=mention_limit))
    if punc_limit > 0:
        funcs.append(partial(limit_punctuations, keep_num=punc_limit))
    if lower_hashtag:
        funcs.append(lower_hashtags)
    if segment_hashtag:
        load()
        funcs.append(segment_hashtags)
    if add_cap_sign:
        funcs.append(add_capital_signs)
    return compose(*funcs)


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    logging.info('Starting preprocessing text with the following parameters:\n{}'.format(
        pprint.pformat(vars(args))))

    logging.info('Creating destination directory {} ...'.format(
        args.output_directory))
    if os.path.exists(args.output_directory):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.mkdir(args.output_directory)

    preprocessor = build_preprocess(demojize=True, textify_emoji=True, mention_limit=3,
                                    punc_limit=3, lower_hashtag=True, segment_hashtag=True, add_cap_sign=True)

    for filename in sorted(glob.glob(os.path.join(args.input_directory, '*.txt'))):
        basename = os.path.basename(filename)
        # logging.debug('Processing filename {} ...'.format(basename))
        with open(filename, mode='rt', encoding='utf-8') as r_fd, open(os.path.join(args.output_directory, basename), mode='wt', encoding='utf-8') as w_fd:
            for line in r_fd:
                processed = preprocessor(line)
                w_fd.write(processed + '\n')

    # logging.info('Finished.')
#
