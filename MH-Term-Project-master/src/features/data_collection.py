"""
---------------------------
This script scraped the 900 posts from social media 'Reddit' belong to 4 classes which are:
    1. depression: 200 posts from depression subreddit
    2. bipolar disorder: 200 posts from bipolar disorder subreddit
    3. PTSD: 200 posts from PTSD subreddit
    4. normal: 300 posts from 7 different subreddits
    How to run the code: python data_collection.py -i 'xyYghCuH5sm9WA' -s 'yMT68q1LgDF1sWE7g_iG5xl8Oik' -u 'jerryhuimanutd' -pw '123456abc!' -o '../' -p 'project'
---------------------------
"""

import os
import praw
import argparse
import pandas as pd
import datetime as dt

def main(arg):
    '''
    Arguments
    ---------
    initilize the information to collect data
    :client_id: string
    :client_secret: string
    :user_agent: string
    :username: string
    :password: string
    '''
    reddit = praw.Reddit(client_id=arg.id,
                         client_secret=arg.secret,
                         user_agent=arg.proj,
                         username=arg.user,
                         password=arg.pw)
                         
    # two dic will store the attributes collected from reddit
    illness_dict = { "title":[], "body":[]}
    normal_dict = { "title":[], "body":[]}
    
    # subreddits we will collect
    subreddit_mentalillness = ['depression','bipolar','ptsd']
    subreddit_normal = ['learnprogramming','movies','politics','fun','nba','finance']
    
    for subr in subreddit_mentalillness:
        subreddit =  reddit.subreddit(subr)
        for submission in subreddit.top(limit=200):
            illness_dict["title"].append(submission.title)
            illness_dict["body"].append(submission.selftext)
    illness_data = pd.DataFrame(illness_dict)
    print len(illness_data)
    illness_data = illness_data[illness_data.body.str.split().map(len)>=10]
    print len(illness_data)

    for subr in subreddit_normal:
        subreddit =  reddit.subreddit(subr)
        for submission in subreddit.top(limit=50):
            normal_dict["title"].append(submission.title)
            normal_dict["body"].append(submission.selftext)
    normal_data = pd.DataFrame(normal_dict)
    print len(normal_data)
    normal_data = normal_data[normal_data.body.str.split().map(len)>=10]
    print len(normal_data)

    # encode two columns to save as csv
    normal_data['body'] = normal_data.body.str.encode('utf-8')
    normal_data['title'] = normal_data.title.str.encode('utf-8')
    illness_data['body'] = illness_data.body.str.encode('utf-8')
    illness_data['title'] = illness_data.title.str.encode('utf-8')

    # save two data frames to csv file
    illness_data.to_csv(arg.outputpath + 'illness_data.csv')
    normal_data.to_csv(arg.outputpath + 'normal_data.csv')

if __name__ == '__main__':
    # pass parameters
    parser = argparse.ArgumentParser(description='This is a data collection for term project in CSCI8360')
    parser.add_argument('-i','--id',required = True)
    parser.add_argument('-s', '--secret', required = True)
    parser.add_argument('-u', '--user', required = True)
    parser.add_argument('-pw', '--pw', required = True)
    parser.add_argument('-o', '--outputpath', required = True)
    parser.add_argument('-p', '--proj', default = 'term project')
    
    args = parser.parse_args()

    main(args)
