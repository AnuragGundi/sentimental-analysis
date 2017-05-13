import pickle

filename = 'data/test/bnaivebayes_test_model.pickle'
infile = open(filename, 'r')        
print pickle.load(infile)        
infile.close()
