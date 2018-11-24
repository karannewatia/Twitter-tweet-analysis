import csv
import nltk
import numpy as np
from sklearn.naive_bayes import MultinomialNB

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

	testTweetDict = {}
	idList = []

	with open('test.csv','r') as testFile:
		testReader = csv.reader(testFile, delimiter = '\n')
		#fields = trainReader.next()
		for i,tweet in enumerate(testReader):
			if not tweet:
				break
			if i == 0:
				continue
			tweet = tweet[0].strip('[').strip(']').split(',')
			id, text, favCount, created, retCount = tweet[0],tweet[1],tweet[3],tweet[5],tweet[12]
			idList.append(id)
			testTweetDict[id] = [text, favCount, created, retCount, words]

	XTe = np.zeros((len(testTweetDict), c+1))
	for i,(tweetId,tweet) in enumerate(testTweetDict.items()):
		tWords = tweet[-1]
		for tw in tWords:
			if tw.lower() in wordDict:
				XTe[i][wordDict[tw.lower()]] = 1
			else:
				XTe[i][c] = 1


	#print(XTe[0])
	
	# indices = np.array(range(len(tweetDict)))
	# np.random.shuffle(indices)
	# XTr = XTr[indices]
	# YTr = YTr[indices]
	# XTr80 = XTr[:int(0.8*len(tweetDict))]
	# YTr80 = YTr[:int(0.8*len(tweetDict))]
	# XTr20 = XTr[int(0.8*len(tweetDict)):]
	# YTr20 = YTr[int(0.8*len(tweetDict)):]

	# clf = MultinomialNB()
	# clf.fit(XTr80, YTr80)
	# preds = clf.predict(XTr20)
	# #print(preds)
	# temp = (np.equal(preds,YTr20))
	#print(np.sum(temp)/temp.shape)

	clf = MultinomialNB()
	clf.fit(XTr, YTr)
	preds = clf.predict(XTe)
	print(preds)

	#print(preds)
	#temp = (np.equal(preds,YTr20))
	#print(np.sum(temp)/temp.shape)

	# with open('output.csv', 'w') as testfile:
	# 	filewriter = csv.writer(testfile, delimiter=',')
	# 	filewriter.writerow(['ID','Label'])
	# 	for i, (id,pred) in enumerate(zip(idList,preds)):
	# 		filewriter.writerow([id,int(pred)])

if __name__ == '__main__':
	main()
