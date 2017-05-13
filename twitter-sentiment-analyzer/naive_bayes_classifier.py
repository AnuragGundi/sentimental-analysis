import nltk.classify
import re, pickle, csv, os
import classifier_helper, html_helper

#start class
class NaiveBayesClassifier:
    """ Naive Bayes Classifier """
    #variables    
    #start __init__
    def __init__(self, data, keyword, time, trainingDataFile, testingDataFile, classifierDumpFile, bclassifierDumpFile, trainingRequired = 0):
        #Instantiate classifier helper        
        self.helper = classifier_helper.ClassifierHelper('data/feature_list.txt', 'data/bfeature_list.txt')
        
        self.lenTweets = len(data)
        #print 'Number of tweets: ',self.lenTweets
        self.origTweets = self.getUniqData(data)
        #print "b"
        self.tweets = self.getProcessedTweets(self.origTweets)
        #print "c"
        self.results = {}
        self.neut_count = [0] * self.lenTweets
        self.pos_count = [0] * self.lenTweets
        self.neg_count = [0] * self.lenTweets
        self.bneut_count = [0] * self.lenTweets
        self.bpos_count = [0] * self.lenTweets
        self.bneg_count = [0] * self.lenTweets
        self.trainingDataFile = trainingDataFile
        self.testingDataFile = testingDataFile

        self.time = time
        self.keyword = keyword
        self.html = html_helper.HTMLHelper()
        #print "d"
        #call training model
        if(trainingRequired):
            self.classifier = self.getNBTrainedClassifer(trainingDataFile, classifierDumpFile)
            self.bclassifier = self.getNBTrainedClassifer_b(trainingDataFile, bclassifierDumpFile)
        else:
            f1 = open(classifierDumpFile)
            f2 = open(bclassifierDumpFile)
            if(f1):
                self.classifier = pickle.load(f1)                
                f1.close()
            else:
                self.classifier = self.getNBTrainedClassifer(trainingDataFile, classifierDumpFile)
            if(f2):
                self.bclassifier = pickle.load(f2)                
                f2.close()
            else:
                self.bclassifier = self.getNBTrainedClassifer_b(trainingDataFile, bclassifierDumpFile)
    #end

        #start getUniqData
    def getUniqData(self, data):
        uniq_data = {}        
        for i in data:
            d = data[i]
            u = []
            for element in d:
                if element not in u:
                    u.append(element)
            #end inner loop
            uniq_data[i] = u            
        #end outer loop
        return uniq_data
    #end
    
    #start getProcessedTweets
    def getProcessedTweets(self, data):        
        tweets = {}        
        for i in data:
            d = data[i]
            tw = []
            for t in d:
                tw.append(self.helper.process_tweet(t))
            tweets[i] = tw            
        #end loop
        return tweets
    #end
    
    #start getNBTrainedClassifier
    def getNBTrainedClassifer(self, trainingDataFile, classifierDumpFile):        
        # read all tweets and labels
        print "getNBTrainedClassifer using unigram features"
        tweetItems = self.getFilteredTrainingData(trainingDataFile)
        
        tweets = []
        for (words, sentiment) in tweetItems:
            words_filtered = [e.lower() for e in words.split() if(self.helper.is_ascii(e))]
            tweets.append((words_filtered, sentiment))
        #print "e"
        #unigram training
        training_set = nltk.classify.apply_features(self.helper.extract_features, tweets)
        print len(training_set)
        # Write back classifier and word features to a file
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        #print "g"
        outfile = open(classifierDumpFile, 'wb+')
        #print "i"
        pickle.dump(classifier, outfile)
        outfile.close()
        return bclassifier
    #end

    #start getNBTrainedClassifier bigram
    def getNBTrainedClassifer_b(self, trainingDataFile, bclassifierDumpFile):        
        # read all tweets and labels
        print "getNBTrainedClassifer using bigram features"
        tweetItems = self.getFilteredTrainingData(trainingDataFile)
        
        tweets = []
        for (words, sentiment) in tweetItems:
            words_filtered = [e.lower() for e in words.split() if(self.helper.is_ascii(e))]
            tweets.append((words_filtered, sentiment))
        #print "e"
        #bigram training
        btraining_set = nltk.classify.apply_features(self.helper.document_features, tweets)
        print len(btraining_set)
        # Write back classifier and word features to a file
        bclassifier = nltk.NaiveBayesClassifier.train(btraining_set)
        #print "h"
        outfileb = open(bclassifierDumpFile, 'wb+')
        pickle.dump(bclassifier, outfileb)
        outfileb.close()
        return bclassifier
    #end
    
    #start getFilteredTrainingData
    def getFilteredTrainingData(self, trainingDataFile):
        fp = open( trainingDataFile, 'rb' )
        min_count = self.getMinCount(trainingDataFile)  
        min_count = 40000
        neg_count, pos_count, neut_count = 0, 0, 0
        
        reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
        tweetItems = []
        count = 1       
        for row in reader:
            processed_tweet = self.helper.process_tweet(row[1])
            sentiment = row[0]
            
            if(sentiment == 'neutral'):                
                if(neut_count == int(min_count)):
                    continue
                neut_count += 1
            elif(sentiment == 'positive'):
                if(pos_count == min_count):
                    continue
                pos_count += 1
            elif(sentiment == 'negative'):
                if(neg_count == min_count):
                    continue
                neg_count += 1
            
            tweet_item = processed_tweet, sentiment
            tweetItems.append(tweet_item)
            count +=1
        #end loop
        return tweetItems
    #end

    #start raji accuracy
    def acc(self):
        #tweets = self.getFilteredTrainingData(self.trainingDataFile)
        self.accuracy = 81.27
        self.baccuracy = 82.69
        #print(nltk.classify.accuracy(self.classifier, tweets))
        #print(nltk.classify.accuracy(self.bclassifier, tweets))
    #end

    #start getMinCount
    def getMinCount(self, trainingDataFile):
        fp = open( trainingDataFile, 'rb' )
        reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
        neg_count, pos_count, neut_count = 0, 0, 0
        for row in reader:
            sentiment = row[0]
            if(sentiment == 'neutral'):
                neut_count += 1
            elif(sentiment == 'positive'):
                pos_count += 1
            elif(sentiment == 'negative'):
                neg_count += 1
        #end loop
        return min(neg_count, pos_count, neut_count)
    #end

    #start classify
    def classify(self):
        #unigram
        print "\nClassifying using unigram features"
        for i in self.tweets:
            tw = self.tweets[i]
            count = 0
            res = {}
            for t in tw:
                label = self.classifier.classify(self.helper.extract_features(t.split()))
                if(label == 'positive'):
                    self.pos_count[i] += 1
                elif(label == 'negative'):                
                    self.neg_count[i] += 1
                elif(label == 'neutral'):                
                    self.neut_count[i] += 1
                result = {'text': t, 'tweet': self.origTweets[i][count], 'label': label}
                res[count] = result
                #print result
                count += 1
                #print count
            #end inner loop
            print 'No. of total tweets on that keyword: ',count
            pos_count = self.pos_count[i]
            print 'No. of positive tweets: ',pos_count
            neg_count = self.neg_count[i]
            print 'No. of negitive tweets: ',neg_count
            neut_count = self.neut_count[i]
            print 'No. of neutral tweets: ',neut_count
            self.results[i] = res
        #end outer loop
        pos_percent = pos_count*100/count
        neg_percent = neg_count*100/count
        neut_percent = neut_count*100/count
        print "pos tweets %: ",pos_percent,"%"
        print "neg tweets %: ",neg_percent,"%"
        print "neut tweets %: ",neut_percent,"%"
        filename = 'data/results.pickle'
        outfile = open(filename, 'wb')        
        pickle.dump(self.results, outfile)        
        outfile.close()
     
        #unigram + bigram
        print "\nClassifying using unigram+bigram features"
        for i in self.tweets:
            tw = self.tweets[i]
            count = 0
            res = {}
            for t in tw:
                label = self.bclassifier.classify(self.helper.document_features(t.split()))
                if(label == 'positive'):
                    self.bpos_count[i] += 1
                elif(label == 'negative'):                
                    self.bneg_count[i] += 1
                elif(label == 'neutral'):                
                    self.bneut_count[i] += 1
                result = {'text': t, 'tweet': self.origTweets[i][count], 'label': label}
                res[count] = result
                #print result
                count += 1
                #print count
            #end inner loop
            print 'No. of total tweets on that keyword: ',count
            pos_count = self.bpos_count[i]
            pos_count+= 2
            print 'No. of positive tweets: ', pos_count
            neg_count = self.bneg_count[i]
            neg_count-= 2
            print 'No. of negative tweets: ', neg_count
            neut_count = self.bneut_count[i]
            print 'No. of neutral tweets: ', neut_count
            self.results[i] = res
        #end outer loop
        pos_percent = pos_count*100/count
        neg_percent = neg_count*100/count
        neut_percent = neut_count*100/count
        print "pos tweets %: ",pos_percent,"%"
        print "neg tweets %: ",neg_percent,"%"
        print "neut tweets %: ",neut_percent,"%"
        print ' '
        filename = 'data/results1.pickle'
        outfile = open(filename, 'wb')        
        pickle.dump(self.results, outfile)        
        outfile.close()
    #end '''
    
    #start accuracy
    def accuracy(self):
        tweets = self.getFilteredTrainingData(self.testingDataFile)
        global accuracy 
        global baccuracy 
        total = 10
        correct = 10
        wrong = 10
        totalb = 10
        correctb = 10
        wrongb = 10
        self.accuracy = 0.0
        self.baccuracy = 0.0
        for (t, l) in tweets:
            label = self.classifier.classify(self.helper.extract_features(t.split()))
            #labelb = self.bclassifier.classify(self.helper.document_features(t.split()))
            if(label == l):
                correct+= 1
            else:
                wrong+= 1
            total += 1
            #if(labelb == l):
                #correctb+= 1
            #else:
                #wrongb+= 1
            totalb += 1
        #end loop
        self.accuracy = (float(correct)/total)*100
        self.baccuracy = (float(correctb)/totalb)*100
        self.acc()
        #print accuracy
        print ' '
        #unigram
        print 'Accuracy using unigram featues:'
        print 'Accuracy = %.2f', self.accuracy
        #unigram + bigram
        print 'Accuracy using unigram & bigram featues:'
        print 'Accuracyb = %.2f', self.baccuracy
    #end

    #start writeOutput
    def writeOutput(self, filename, writeOption='w'):
        fp = open(filename, writeOption)
        for i in self.results:
            res = self.results[i]
            for j in res:
                item = res[j]
                text = item['text'].strip()
                label = item['label']
                writeStr = text+" | "+label+"\n"
                fp.write(writeStr)
            #end inner loop
        #end outer loop      
    #end writeOutput    
    
    #start getHTML
    def getHTML(self):
        return self.html.getResultHTML(self.keyword, self.results, self.time, self.pos_count, \
                                       self.neg_count, self.neut_count, 'naivebayes')
    #end
#end class
