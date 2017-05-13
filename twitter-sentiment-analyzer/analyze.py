import get_twitter_data
import baseline_classifier, naive_bayes_classifier, max_entropy_classifier#, libsvm_classifier
import json,sys,pickle

#keyword = 'arnab'
time = 'today'
if(len(sys.argv) < 2):
    print "Please choose the algorithm to test, sytanx = python analyze.py (svm|naivebayes|maxent)"
    exit() 
keyword = str(sys.argv[1])
print 'Your keyword is :', keyword
print 'Getting Twitter Data with keyword ',keyword
twitterData = get_twitter_data.TwitterData()
tweets = twitterData.getTwitterData(keyword, time)
print 'Got Twitter Data'
print ' '
#print tweets
#print length(tweets)
#algorithm = 'baseline'
#algorithm = 'naivebayes'
#algorithm = 'maxent'
#algorithm = 'svm'
    
algorithm = 'naivebayes'

if(algorithm == 'baseline'):
    bc = baseline_classifier.BaselineClassifier(tweets, keyword, time)
    bc.classify()
    #val = bc.getHTML()
    filename = 'data/results_lastweek.pickle'
    infile = open(filename, 'r')        
    print pickle.load(infile)        
    infile.close()
elif(algorithm == 'naivebayes'):
    print 'naivebayes started'
    #trainingDataFile = 'data/training_trimmed.csv'
    trainingDataFile = 'data/full_training_dataset.csv'
    testingDataFile = 'data/sampleTweets.csv'
    classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
    bclassifierDumpFile = 'data/test/naivebayes_test_model_b.pickle'
    trainingRequired = 0
    nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time,\
                                  trainingDataFile, testingDataFile, classifierDumpFile, bclassifierDumpFile, trainingRequired)
    #print '2'
    nb.classify()
    print 'naivebayes finished'
    filename = 'data/results.pickle'
    infile = open(filename, 'r')        
    #print pickle.load(infile)        
    infile.close()
    #print '3'
    nb.accuracy()
    #nb.acc()
    #print '4'
elif(algorithm == 'maxent'):    
    #trainingDataFile = 'data/training_trimmed.csv'
    trainingDataFile = 'data/full_training_dataset.csv'
    classifierDumpFile = 'data/test/maxent_test_model.pickle'
    trainingRequired = 1
    maxent = max_entropy_classifier.MaxEntClassifier(tweets, keyword, time,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
    #maxent.analyzeTweets()
    maxent.classify()
    maxent.accuracy()
elif(algorithm == 'svm'):    
    trainingDataFile = 'data/training_trimmed.csv'
    #trainingDataFile = 'data/full_training_dataset.csv'                
    classifierDumpFile = 'data/test/svm_test_model.pickle'
    trainingRequired = 1
    sc = libsvm_classifier.SVMClassifier(tweets, keyword, time,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
    sc.classify()
    sc.accuracy()

print 'THE END'
