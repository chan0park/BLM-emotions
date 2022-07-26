""" Code to massively download Twitter data. It defaults to authenticate at
    Twitter through application-only authentication [1].

References:
    [1] https://developer.twitter.com/en/docs/basics/authentication/overview/application-only
    [2] https://developer.twitter.com/en/docs/basics/authentication/api-reference/token
    [3] https://developer.twitter.com/en/docs/basics/rate-limiting
    [4] https://developer.twitter.com/en/docs/basics/rate-limits
    [5] https://developer.twitter.com/en/docs/developer-utilities/rate-limit-status/api-reference/get-application-rate_limit_status
    [6] https://developer.twitter.com/en/docs/basics/response-codes
    [7] https://developer.twitter.com/en/docs/tweets/search/overview
    [8] https://developer.twitter.com/en/docs/tweets/search/guides/build-standard-query
    [9] https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators
    [10] https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
    [11] https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/intro-to-tweet-json
    [12] https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
    [13] https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/user-object
    [14] https://developer.twitter.com/en/docs/accounts-and-users/follow-search-get-users/api-reference/get-users-show
    [15] https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
    [16] https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline
    [17] https://developer.twitter.com/en/docs/tweets/tweet-updates.html
    [18] https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-retweeters-ids
    [19] https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-lookup
    [20] https://developer.twitter.com/en/docs/twitter-api/v1/accounts-and-users/follow-search-get-users/api-reference/get-users-lookup
"""


import logging
import http
import http.client
import traceback
import base64
import urllib
import json
import time
import os

import requests_oauthlib
import requests


class TwitterUserNotFoundException(Exception):
    pass


class TwitterUserSuspendedException(Exception):
    pass


class ProtectedTweetsException(Exception):
    pass


class TwitterServerErrorException(Exception):
    pass


class TwitterReader:


    ##### PRIVATE CLASS MEMBERS #####


    _endpoint                       = 'api.twitter.com'
    _connection                     = None
    _debug_connection               = None
    _request_headers                = None

    _app_name                       = None
    _consumer_key                   = None
    _consumer_secret                = None

    _user_auth                      = None
    _connection_oauth1              = None
    _access_token                   = None
    _access_token_secret            = None


    _limits = {                                                         # Dictionary containing resources rate limits information. The key represents a resource that contains two associated keys: remaining = how many requests are left for the resource; renew_epoch = next epoch to renew the window for the resource.
               '/users/show'            : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/users/lookup'          : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/statuses/user_timeline': {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/search/tweets'         : {
                                            'remaining'     : None,
                                            'renew_epoch'   : None,
                                          },
               '/statuses/retweeters'   : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/statuses/lookup'       : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/friends/list'          : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/followers/list'        : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
               '/friendships/show'      : {
                                           'remaining'      : None,
                                           'renew_epoch'    : None,
                                          },
              }

    _logger                         = None


    def __init__(self, app_name, consumer_key, consumer_secret, user_auth=False, access_token=None, access_token_secret=None, debug_connection=False):
        self._app_name = app_name
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._user_auth = user_auth
        if self._user_auth:
            self._access_token = access_token
            self._access_token_secret = access_token_secret
        self._debug_connection = debug_connection

        # limits set to 1 to allow the first request, after then the values are updated from Twitter headers
        self._limits['/users/show']['remaining'] = 1
        self._limits['/users/lookup']['remaining'] = 1
        self._limits['/statuses/user_timeline']['remaining'] = 1
        self._limits['/search/tweets']['remaining'] = 1
        self._limits['/statuses/retweeters']['remaining'] = 1
        self._limits['/statuses/lookup']['remaining'] = 1
        self._limits['/friends/list']['remaining'] = 1
        self._limits['/followers/list']['remaining'] = 1
        self._limits['/friendships/show']['remaining'] = 1

        self._logger = logging.getLogger(self.__class__.__name__)


    def _handle_twitter_response_code(self, response, data, user_id = ''):
        if (self._user_auth and response.status_code == requests.codes.OK) or \
           (not self._user_auth and response.status == http.HTTPStatus.OK):
            return
        try:
            twitter_error = json.loads(data, encoding='utf-8')['errors'][0]
            twitter_error_msg = ''.join(['Twitter error message: ', str(twitter_error['code']), ' - ', twitter_error['message']])
        except Exception as e:
            twitter_error_msg = '(empty or invalid Twitter error message)'
        if self._user_auth:
            error_msg = ''.join(['HTTP error code: ', str(response.status_code), '. ', twitter_error_msg])
        else:
            error_msg = ''.join(['HTTP error message: ', str(response.status), ' - ', response.reason, '. ', twitter_error_msg])
        if (self._user_auth and response.status_code == requests.codes.NOT_FOUND) or \
           (not self._user_auth and response.status == http.HTTPStatus.NOT_FOUND):      # inexistent user, maybe has got himself out from Twitter
            raise TwitterUserNotFoundException(''.join(['User id = ', user_id, ' not found. ', error_msg]))
        elif (self._user_auth and response.status_code == requests.codes.FORBIDDEN) or \
             (not self._user_auth and response.status == http.HTTPStatus.FORBIDDEN):    # suspended user
            raise TwitterUserSuspendedException(''.join(['User id = ', user_id, ' suspended. ', error_msg]))
        elif (self._user_auth and response.status_code == requests.codes.UNAUTHORIZED) or \
             (not self._user_auth and response.status == http.HTTPStatus.UNAUTHORIZED): # protected tweet
            raise ProtectedTweetsException(''.join(['Tweets from user id = ', user_id, ' are protected. ', error_msg]))
        elif (self._user_auth and (response.status_code // 100) == 5) or \
             (not self._user_auth and (response.status // 100) == 5):                   # HTTP server error
            raise TwitterServerErrorException('HTTP server error. ' + error_msg)
        else:
            raise Exception(error_msg)


    def _get_request_headers(self):
        consumer_cred_base64 = base64.b64encode(bytes(self._consumer_key + ':' + self._consumer_secret, 'ascii'))
        bearer_token_headers = {'Host': self._endpoint,
                                'User-Agent': self._app_name,
                                'Authorization': 'Basic ' + str(consumer_cred_base64, 'ascii'),
                                'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                                #'Accept-Encoding': 'gzip',     # gives error
                               }
        bearer_token_params = urllib.parse.urlencode({'grant_type': 'client_credentials'})
        self._connection.request('POST', '/oauth2/token', headers=bearer_token_headers, body=bearer_token_params)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data)
        bearer_token_dict = json.loads(data, encoding='utf-8')
        if ('token_type' not in bearer_token_dict) or (bearer_token_dict['token_type'] != 'bearer'):
            raise Exception(''.join(['Invalid JSON response from Twitter : ', str(bearer_token_dict)]))
        self._request_headers = { 'Host': self._endpoint ,
                                  'User-Agent': self._app_name,
                                  'Authorization': 'Bearer ' + bearer_token_dict['access_token'],
                                  'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                                  #'Accept-Encoding': 'gzip',     # gives error
                                }


    def _get_rate_limit_status(self, resource):
        family = resource.split('/')[1]
        rate_limit_url = '/1.1/application/rate_limit_status.json'
        params = { 'resources' : family }
        encoded_params = '?%s' % urllib.parse.urlencode(params)
        retry = True
        while retry:
            self._logger.debug(''.join(['Absent rate limit headers. Requesting rate limits for resource family ', family , ' ...']))
            try:
                if self._user_auth:
                    response = self._connection_oauth1.get(''.join(['https://', self._endpoint, rate_limit_url, encoded_params]))
                    data = response.text
                else:
                    self._connection.request('GET', rate_limit_url + encoded_params, headers=self._request_headers)
                    response = self._connection.getresponse()
                    data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
                response_key = resource + '/:id' if resource == '/users/show' else resource
                self._handle_twitter_response_code(response, data)
                limits = json.loads(data, encoding='utf-8')
                self._limits[resource]['remaining'] = limits['resources'][family][response_key]['remaining']
                self._limits[resource]['renew_epoch'] = limits['resources'][family][response_key]['reset']
                retry = False
            except Exception as e:
                self._logger.warning(''.join(['Error requesting rate limits for resource family ', family , ' . Error: ', str(e), ' Sleeping 5 seconds and retrying ...']))
                time.sleep(5)       # this 'application/rate_limit_status' resource can be queried 180 times in a 15-minutes window (at each 5 seconds)


    def _check_limit_remaining(self, resource):
        if self._limits[resource]['remaining'] <= 0:
            if self._limits[resource]['remaining'] == -1:       # Twitter didn't send the rate limits headers, request these limits
                self._get_rate_limit_status(resource)
            if self._limits[resource]['remaining'] == 0:
                sleep_sec = (self._limits[resource]['renew_epoch'] + 1) - time.time() # (renew_epoch + 1) => avoiding synchonization problems
                sleep_sec = 0 if sleep_sec < 0 else sleep_sec
                self._logger.warning(''.join(['Requests limit reached. Sleeping for ', str(sleep_sec), ' seconds ...']))
                time.sleep(sleep_sec)
                self.reconnect()    # better to force a restart since the server maybe had dropped the current connection


    def _update_rate_limit(self, resource, response):
        # headers can be absent
        if self._user_auth:
            self._limits[resource]['remaining'] = int(response.headers.get('x-rate-limit-remaining', -1))
            self._limits[resource]['renew_epoch'] = int(response.headers.get('x-rate-limit-reset', -1))
        else:
            self._limits[resource]['remaining'] = int(response.getheader('x-rate-limit-remaining', default='-1'))
            self._limits[resource]['renew_epoch'] = int(response.getheader('x-rate-limit-reset', default='-1'))


    def _request_tweets(self, params):
        self._check_limit_remaining('/statuses/user_timeline')
        encoded_params = '?%s' % urllib.parse.urlencode(params)
        self._connection.request('GET', '/1.1/statuses/user_timeline.json' + encoded_params, headers=self._request_headers)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data, params['user_id'])
        self._update_rate_limit('/statuses/user_timeline', response)
        return json.loads(data, encoding='utf-8')


    ##### PUBLIC CLASS MEMBERS #####


    def connect(self):
        self._logger.debug(''.join(['Connecting to Twitter endpoint ', self._endpoint, ' ...']))

        if self._user_auth:
            self._connection_oauth1 = requests_oauthlib.OAuth1Session(client_key=self._consumer_key,
                                                                     client_secret=self._consumer_secret,
                                                                     resource_owner_key=self._access_token,
                                                                     resource_owner_secret=self._access_token_secret,
                                                                    )
            logging.getLogger('urllib3').setLevel(logging.INFO)
            logging.getLogger('oauthlib').setLevel(logging.INFO)
            logging.getLogger('requests_oauthlib').setLevel(logging.INFO)
        else:
            self._connection = http.client.HTTPSConnection(self._endpoint)
            self._connection.set_debuglevel(1 if self._debug_connection else 0)
            if not self._request_headers:
                self._logger.debug('Trying to get application bearer token ...')
                self._get_request_headers()


    def cleanup(self):
        if not self._user_auth:
            self._connection.close()


    def reconnect(self):
        if not self._user_auth:
            self._logger.info(''.join(['Restarting connection to Twitter endpoint ', self._endpoint, ' ...']))
            try:
                self._connection.close()
            except Exception as e:
                self._logger.warning('Error trying to close twitter connection while reconnecting.')
                traceback.print_exc()
            self.connect()


    def search_users(self, word, language = 'en', max_results = 1000):
        """ Return a list of user based on a list of words.

        Besides using the word parameter to filter the language of the user, the
        code also uses the parameter 'language' in the Twitter search API. The
        query option '-filter:retweets' is also used in the search API to filter
        out retweets. It is recommended not to use too small function words
        (less than 3 characters) since they can match undesired languages.

        Peudo-code:
            read list of words
            for each 'word':
                search most recents tweets using 'word' and recover at most max_results_per_word parameter
            for each 'tweet':
                save user information if not already seen
        """
        search_params = {'q':                   word + ' -filter:retweets',
                         'lang' :               language,
                         'result_type' :        'recent',
                         'count' :              100,
                         'include_entities' :   'true',
                 }
        encoded_search_params = '?%s' % urllib.parse.urlencode(search_params)
        acc_results = 0
        users = {}
        while acc_results < max_results:
            # get tweets
            self._check_limit_remaining('/search/tweets')
            self._connection.request('GET', '/1.1/search/tweets.json' + encoded_search_params, headers=self._request_headers)
            response = self._connection.getresponse()
            data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
            self._handle_twitter_response_code(response, data)
            tweets = json.loads(data, encoding='utf-8')
            self._update_rate_limit('/search/tweets', response)

            # find users
            for tweet in tweets['statuses']:
                if tweet['user']['id_str'] in users:
                    self._logger.debug(''.join([ '\t\tUser ', tweet['user']['screen_name'], ' , id = ', tweet['user']['id_str'], ' already seen. Ignoring ...' ]))
                    continue
                self._logger.debug(''.join([ '\t\tAdding user ', tweet['user']['screen_name'], ' , id = ', tweet['user']['id_str'] ]))
                users[tweet['user']['id_str']] = tweet['user']

            # account results
            results = len(tweets['statuses'])
            acc_results += results
            self._logger.debug(''.join(['\tRetrieved ', str(results), ' tweets. Current number of users found = ',  str(len(users.keys())), '. Remaining \'/search/tweets\' requests = ', str(self._limits['/search/tweets']['remaining']), '.']))

            # get next results page
            if 'next_results' not in tweets['search_metadata']:     # end of results
                break
            encoded_search_params = tweets['search_metadata']['next_results']

        self._logger.debug(''.join(['Number of users found for word ', word, ' = ',  str(len(users.keys())), '.']))
        return users


    def get_user_info(self, user_id):
        self._check_limit_remaining('/users/show')
        self._connection.request('GET', ''.join(['/1.1/users/show.json?user_id=', user_id]), headers=self._request_headers)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data, user_id)
        self._update_rate_limit('/users/show', response)
        self._logger.debug(''.join(['Remaining \'/users/show\' requests = ', str(self._limits['/users/show']['remaining']), '.']))
        return json.loads(data, encoding='utf-8')


    def get_users_info(self, user_ids):
        users = []
        lookup_url = '/1.1/users/lookup.json'
        lookup_url_key = '/users/lookup'
        max_number_ids_allowed = 100
        for idx in range(0, len(user_ids), max_number_ids_allowed):
            users_step = ','.join(user_ids[idx: idx+max_number_ids_allowed])
            params = {'user_id': users_step}
            self._check_limit_remaining(lookup_url_key)
            if self._user_auth:
                response = self._connection_oauth1.post(''.join(['https://', self._endpoint, lookup_url]), data=params)
                data = response.text
            else:
                #self._connection.request('POST', lookup_url, headers=self._request_headers, body=urllib.parse.urlencode(params))   # Despite being recommended by the API, rate limiting does not work invoking this endpoint through POST (less allowed requests and lack of rate limit headers)
                params_encoded = '?%s' % urllib.parse.urlencode(params)
                self._connection.request('GET', lookup_url + params_encoded, headers=self._request_headers)
                response = self._connection.getresponse()
                data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
            self._handle_twitter_response_code(response, data)
            self._update_rate_limit(lookup_url_key, response)
            self._logger.debug('Remaining \'{}\' requests = {} .'.format(lookup_url_key, self._limits[lookup_url_key]['remaining']))
            users += json.loads(data, encoding='utf-8')
        return users


    def get_user_timeline(self, user_id, since_id=None, extended=True):
        """ Downloads all the tweets in the user timeline according to [15].
        """
        timeline_url = '/1.1/statuses/user_timeline.json'
        timeline_params = {'user_id'            : user_id,
                           'count'              : 200,
                           'include_rts'        : 'true',
                           'exclude_replies'    : 'false',
                           'trim_user'          : 'true',
                          }

        if since_id:
            self._logger.debug('Retrieving tweets since id {} ...'.format(since_id))
            timeline_params['since_id'] = int(since_id)

        if extended:    # extended tweets format [17]
            self._logger.debug('Retrieving extended tweets (more than 140 characters) ...')
            timeline_params['tweet_mode'] = 'extended'

        # first timeline request
        tweets = self._request_tweets(timeline_params)
        retrieved_tweets = len(tweets)
        self._logger.debug(''.join(['Retrieved ', str(retrieved_tweets), ' tweets in the first request. Remaining \'/statuses/user_timeline\' requests = ', str(self._limits['/statuses/user_timeline']['remaining']), '.']))
        if retrieved_tweets == 0:    # finish this profile collecting
            return tweets

        # older tweets
        while retrieved_tweets > 0:
            timeline_params['max_id'] = tweets[-1]['id'] - 1
            temp = self._request_tweets(timeline_params)
            retrieved_tweets = len(temp)
            self._logger.debug(''.join(['Retrieved ', str(retrieved_tweets), ' tweets. Remaining \'/statuses/user_timeline\' requests = ', str(self._limits['/statuses/user_timeline']['remaining']), '.']))
            tweets += temp
        del timeline_params['max_id']

        # newer tweets since collecting
        timeline_params['since_id'] = tweets[0]['id']
        temp = self._request_tweets(timeline_params)
        self._logger.debug(''.join(['Retrieved ', str(len(temp)), ' newer tweets since collecting. Remaining \'/statuses/user_timeline\' requests = ', str(self._limits['/statuses/user_timeline']['remaining']), '.']))
        return temp + tweets


    def search_expression(self,
                          expr,
                          language='en',
                          max_results=1000,
                          extended=True,
                          retweets=False,
                          since_id=None,
                          max_id=None,
                          until=None,
                          fd=None,
                         ):
        """ Downloads tweets that contains a specific expression.
        """
        if max_results == 0:
            max_results = float('inf')
        search_params = {'q':                   '\"{}\"'.format(expr),
                         'lang' :               language,
                         'result_type' :        'recent',
                         'count' :              100,
                         'include_entities' :   'true',
                        }
        additional_search_params = {}   # parameters not retrieved in the search_metadata.next_results field
        if extended:    # extended tweets format [17]
            self._logger.debug('Retrieving extended tweets (more than 140 characters) ...')
            search_params['tweet_mode'] = 'extended'
            additional_search_params['tweet_mode'] = 'extended'
        if not retweets:
            self._logger.debug('Filtering out retweets ...')
            search_params['q'] += ' -filter:retweets'
        if since_id:
            self._logger.debug('Retrieving tweets newer than tweet ID {} ...'.format(since_id))
            search_params['since_id'] = since_id
            additional_search_params['since_id'] = since_id
        if max_id:
            self._logger.debug('Retrieving tweets older than tweet ID {} ...'.format(max_id))
            search_params['max_id'] = max_id
        if until:
            self._logger.debug('Retrieving tweets older than date {} ...'.format(until))
            search_params['until'] = until
        encoded_search_params = '?%s' % urllib.parse.urlencode(search_params)
        encoded_additional_search_params = '&%s' % urllib.parse.urlencode(additional_search_params)

        acc_results = 0
        total_tweets = []
        if fd:
            first_tweet = True
            fd.write('[')
        while acc_results < max_results:
            # get tweets
            self._check_limit_remaining('/search/tweets')
            self._connection.request('GET', '/1.1/search/tweets.json' + encoded_search_params, headers=self._request_headers)
            response = self._connection.getresponse()
            data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
            self._handle_twitter_response_code(response, data)
            tweets = json.loads(data, encoding='utf-8')
            self._update_rate_limit('/search/tweets', response)

            if tweets['statuses']:
                if fd:  # to avoid memory exaustion
                    if not first_tweet:
                        fd.write(',')
                    for tweet in tweets['statuses'][:-1]:
                        fd.write('{},'.format(json.dumps(tweet, sort_keys=True, ensure_ascii=True)))
                    for tweet in tweets['statuses'][-1:]:
                        fd.write('{}'.format(json.dumps(tweet, sort_keys=True, ensure_ascii=True)))
                    first_tweet = False
                else:
                    total_tweets += tweets['statuses']

            # account results
            results = len(tweets['statuses'])
            acc_results += results
            self._logger.debug('\tRetrieved {} tweets. Current number of tweets found = {}. Remaining \'/search/tweets\' requests = {}.'.format(results, acc_results, self._limits['/search/tweets']['remaining']))
            # get next results page
            if 'next_results' not in tweets['search_metadata']:     # end of results
                break
            encoded_search_params = tweets['search_metadata']['next_results']
            encoded_search_params += encoded_additional_search_params

        if fd:
            fd.write(']')
        self._logger.debug('Number of tweets found for expr \'{}\' = {}'.format(expr, acc_results))
        return total_tweets


    def hydrate_tweets(self, tweet_ids, extended=True):
        """ Retrieves tweets (hydrate) from their tweet ids [19].
        """

        lookup_url = '/1.1/statuses/lookup.json'
        lookup_url_key = '/statuses/lookup'
        params = {'id'                  : None,
                  'include_entities'    : 'false',
                  'trim_user'           : 'true',
                  'map'                 : 'false',
                  'include_ext_alt_text': 'false',
                  'include_card_uri'    : 'false',
                 }
        if extended:    # extended tweets format [17]
            self._logger.debug('Retrieving extended tweets (more than 140 characters) ...')
            params['tweet_mode'] = 'extended'
        max_number_ids_allowed = 100

        total_tweets = []
        for idx in range(0, len(tweet_ids), max_number_ids_allowed):
            params['id'] = ','.join(tweet_ids[idx: idx+max_number_ids_allowed])
            self._check_limit_remaining(lookup_url_key)
            if self._user_auth:
                response = self._connection_oauth1.post(''.join(['https://', self._endpoint, lookup_url]), data=params)
                data = response.text
            else:
                #self._connection.request('POST', lookup_url, headers=self._request_headers, body=urllib.parse.urlencode(params))   # Despite being recommended by the API, rate limiting does not work invoking this endpoint through POST (less allowed requests and lack of rate limit headers)
                params_encoded = '?%s' % urllib.parse.urlencode(params)
                self._connection.request('GET', lookup_url + params_encoded, headers=self._request_headers)
                response = self._connection.getresponse()
                data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse

            self._update_rate_limit(lookup_url_key, response)
            self._handle_twitter_response_code(response, data)
            tweets = json.loads(data, encoding='utf-8')
            total_tweets += tweets
            self._logger.debug('\tRetrieved {} tweets. Current number of tweets retrieved = {}. Remaining \'{}\' requests = {}.'.format(len(tweets), len(total_tweets), lookup_url_key, self._limits[lookup_url_key]['remaining']))

        self._logger.debug('Total number of tweets retrieved {}/{}.'.format(len(total_tweets), len(tweet_ids)))
        return total_tweets


    def get_retweeters(self, tweet_id):
        retweet_url = '/1.1/statuses/retweeters/ids.json'
        retweet_params = {'id'      : tweet_id,
                           'count'  : 100,
                           'cursor' : -1,
                          }
        self._check_limit_remaining('/statuses/retweeters')
        encoded_params = '?%s' % urllib.parse.urlencode(retweet_params)
        self._connection.request('GET', retweet_url + encoded_params, headers=self._request_headers)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data)
        self._update_rate_limit('/statuses/retweeters', response)
        self._logger.debug(''.join(['Remaining \'/statuses/retweeters\' requests = ', str(self._limits['/statuses/retweeters']['remaining']), '.']))
        data = json.loads(data, encoding='utf-8')
        retweeters = data['ids'][:]
        while data['next_cursor'] != 0:
            retweet_params['cursor'] = data['next_cursor']
            self._check_limit_remaining('/statuses/retweeters')
            encoded_params = '?%s' % urllib.parse.urlencode(retweet_params)
            self._connection.request('GET', retweet_url + encoded_params, headers=self._request_headers)
            response = self._connection.getresponse()
            data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
            self._handle_twitter_response_code(response, data)
            self._update_rate_limit('/statuses/retweeters', response)
            self._logger.debug(''.join(['Remaining \'/statuses/retweeters\' requests = ', str(self._limits['/statuses/retweeters']['remaining']), '.']))
            data = json.loads(data, encoding='utf-8')
            retweeters += data['ids'][:]
        return retweeters


    def get_friends(self, screen_name):
        friends_url = '/1.1/friends/list.json'
        friends_params = {'screen_name'             : screen_name,
                          'count'                   : 200,
                          'cursor'                  : -1,
                          'skip_status'             : True,
                          'include_user_entities'   : False,
                          }
        self._check_limit_remaining('/friends/list')
        encoded_params = '?%s' % urllib.parse.urlencode(friends_params)
        self._connection.request('GET', friends_url + encoded_params, headers=self._request_headers)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data)
        self._update_rate_limit('/friends/list', response)
        self._logger.debug(''.join(['Remaining \'/friends/list\' requests = ', str(self._limits['/friends/list']['remaining']), '.']))
        data = json.loads(data, encoding='utf-8')
        friends = data['users'][:]
        while data['next_cursor'] != 0:
            friends_params['cursor'] = data['next_cursor']
            self._check_limit_remaining('/friends/list')
            encoded_params = '?%s' % urllib.parse.urlencode(friends_params)
            self._connection.request('GET', friends_url + encoded_params, headers=self._request_headers)
            response = self._connection.getresponse()
            data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
            self._handle_twitter_response_code(response, data)
            self._update_rate_limit('/friends/list', response)
            self._logger.debug(''.join(['Remaining \'/friends/list\' requests = ', str(self._limits['/friends/list']['remaining']), '.']))
            data = json.loads(data, encoding='utf-8')
            friends += data['users'][:]
        return friends


    def get_followers(self, screen_name):
        followers_url = '/1.1/followers/list.json'
        followers_params = {'screen_name'           : screen_name,
                            'count'                 : 200,
                            'cursor'                : -1,
                            'skip_status'           : True,
                            'include_user_entities' : False,
                          }
        self._check_limit_remaining('/followers/list')
        encoded_params = '?%s' % urllib.parse.urlencode(followers_params)
        self._connection.request('GET', followers_url + encoded_params, headers=self._request_headers)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data)
        self._update_rate_limit('/followers/list', response)
        self._logger.debug(''.join(['Remaining \'/followers/list\' requests = ', str(self._limits['/followers/list']['remaining']), '.']))
        data = json.loads(data, encoding='utf-8')
        followers = data['users'][:]
        while data['next_cursor'] != 0:
            followers_params['cursor'] = data['next_cursor']
            self._check_limit_remaining('/followers/list')
            encoded_params = '?%s' % urllib.parse.urlencode(followers_params)
            self._connection.request('GET', followers_url + encoded_params, headers=self._request_headers)
            response = self._connection.getresponse()
            data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
            self._handle_twitter_response_code(response, data)
            self._update_rate_limit('/followers/list', response)
            self._logger.debug(''.join(['Remaining \'/followers/list\' requests = ', str(self._limits['/followers/list']['remaining']), '.']))
            data = json.loads(data, encoding='utf-8')
            followers += data['users'][:]
        return followers


    def get_friendship(self, source_screen_name, target_screen_name):
        friendship_url = '/1.1/friendships/show.json'
        friendship_params = {'source_screen_name'   : source_screen_name,
                             'target_screen_name'   : target_screen_name,
                            }
        self._check_limit_remaining('/friendships/show')
        encoded_params = '?%s' % urllib.parse.urlencode(friendship_params)
        self._connection.request('GET', friendship_url + encoded_params, headers=self._request_headers)
        response = self._connection.getresponse()
        data = response.read().decode('utf-8')  # See note on https://docs.python.org/2/library/httplib.html#httplib.HTTPConnection.getresponse
        self._handle_twitter_response_code(response, data)
        self._update_rate_limit('/friendships/show', response)
        self._logger.debug(''.join(['Remaining \'/friendship/show\' requests = ', str(self._limits['/friendships/show']['remaining']), '.']))
        return json.loads(data, encoding='utf-8')
