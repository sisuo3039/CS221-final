import urllib2 as ul2
import json
class twitter:
    def __init__(self,fromDate,toDate,lang='en'):
        self.baseURL='https://xs01sakp.us1.hana.ondemand.com/sap/rds-bdi/semantic_V4/authorization/services.xsodata'
        self.lang=lang
        self.fromDate=fromDate
        self.toDate=toDate
    def getDateRangeQuery(self):
        daterange="created ge datetime%27"+ self.fromDate +"T00:00:00%27 and created le datetime%27"+ self.toDate+"T23:59:59%27"

        return daterange
    def getLangQuery(self):
        return "lang eq %27"+ self.lang+"%27"

    def getRetweetedQuery(self):
        return "retweetedUser ne %27%27"

    def getReplyQuery(self):
        return "replyUser ne %27%27"

    def getOriginalQuery(self):
        return 'replyUser eq %27%27 and retweetedUser eq %27%27'

    def request(self,query):
        # url= self.baseURL+ul.quote(query)
        url=self.baseURL+query.replace(' ','%20')
        response=ul2.urlopen(url)
        return  response.read()

    def getPostCount(self,param='all'):
        if param == 'reply':
            query = '/Tweets/$count?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() +' and '+self.getReplyQuery()+')'
        elif param =='retweet':
            query = '/Tweets/$count?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() + ' and ' + self.getRetweetedQuery() + ')'
        elif param == 'original':
            query = '/Tweets/$count?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() + ' and ' + self.getOriginalQuery() + ')'
        else:
            query='/Tweets/$count?$filter=('+self.getDateRangeQuery()+' and ' + self.getLangQuery()+')'
        data=self.request(query)
        print ('getPostCount from date %s , to date %s , language = %s is -> %s <-' %(self.fromDate,self.toDate,self.lang,data))
        return data

    def getSentimentSum(self,param='all'):
        if param== 'reply':
            query = '/Tweets?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() + ' and '+self.getReplyQuery()+')&$select=sentiment&$format=json'
        elif param =='retweet':
            query = '/Tweets?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() + ' and '+self.getRetweetedQuery()+')&$select=sentiment&$format=json'
        elif param =='original':
            query = '/Tweets?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() + ' and '+self.getOriginalQuery()+')&$select=sentiment&$format=json'
        else:
            query = '/Tweets?$filter=(' + self.getDateRangeQuery() + ' and ' + self.getLangQuery() + ')&$select=sentiment&$format=json'
        data=json.loads(self.request(query))
        sentimentV=data['d']['results'][0]['sentiment']
        sentimentV=[0,float(sentimentV)][sentimentV<>'null']

        print ('getSentimentSum from date %s , to date %s , language = %s is -> %s <-' %(self.fromDate,self.toDate,self.lang,sentimentV))
        return sentimentV

    def getAvgSentiment(self,param='all'):
        sentimentSum=self.getSentimentSum(param)
        sentimentCount=float(self.getPostCount(param))
        avgSentiment=sentimentSum/sentimentCount
        print ('getAvgSentiment from date %s , to date %s , language = %s is -> %s <-' %(self.fromDate,self.toDate,self.lang,avgSentiment))
        return sentimentSum,sentimentCount,avgSentiment


#
# Twitter class encapsulate all the twitter data range from 2015 till today for keyword "SAP" around 13M
# For now the sentiment classification is still running for language = 'en' and
# once finish will process remaining supported languages
#
# to use this class , instantiate with from data, to today ,default lang='en'
#
# eg:
#                tData= twitter('2016-10-01','2017-11-09')
#                which get data from 2016-10-01 00.00.00 to 2017-11-08 23.59.59
#
# There are three main function getSentimentSum,getPostCount,getAvgSentiment .
# all functions will take param  which can have value :  reply,retweet,original,all (default to all)
# eg:
#               tData.getAvgSentiment(param='original')
#        getAvgSentiment will return (sentimentSum,sentimentCount,avgSentiment)
# which return the sum,count and avg sentiment for only the original post
#
#
if __name__ == "__main__":
    result=twitter('2017-06-01','2017-11-09').getAvgSentiment()

    print result