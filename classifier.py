import csv
import nltk
import numpy as np

def main():

	tweetDict = {}
	wordDict = {}
	c = 0

	with open('train.csv','r') as trainFile:
		trainReader = csv.reader(trainFile, delimiter = '\n')
		#fields = trainReader.next()
		for i,tweet in enumerate(trainReader):
			if not tweet:
				break
			if i == 0:
				continue
			tweet = tweet[0].strip('[').strip(']').split(',')
			id, text, favCount, created, retCount, label = tweet[0],tweet[1],tweet[3],tweet[5],tweet[12],tweet[17]
			words = nltk.word_tokenize(text)
			for w in words:
				if not(w.lower() in wordDict):
					wordDict[w.lower()] = c 
					c += 1
			tweetDict[id] = [text, favCount, created, retCount, label, words]

	XTr = np.zeros((len(tweetDict), c+1))
	YTr = np.zeros(len(tweetDict))
	for i,(tweetId,tweet) in enumerate(tweetDict.items()):
		tWords = tweet[-1]
		YTr[i] = tweet[-2]
		for tw in tWords:
			if tw.lower() in wordDict:
				XTr[i][wordDict[tw.lower()]] = 1 
			else:
				XTr[i][c] = 1

	print(XTr)
	print(XTr.shape, len(tweetDict),c)
	print(YTr)
	print(YTr.shape)
		 

if __name__ == '__main__':
	main()