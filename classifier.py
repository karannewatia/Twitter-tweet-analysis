import csv
import nltk
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import RidgeClassifierCV

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

	XTr = np.zeros((len(tweetDict), c+11))
	YTr = np.zeros(len(tweetDict))
	for i,(tweetId,tweet) in enumerate(tweetDict.items()):
		tWords = tweet[-1]
		YTr[i] = tweet[-2]
		for tw in tWords:
			if tw.lower() in wordDict:
				XTr[i][wordDict[tw.lower()]] += 1
			else:
				XTr[i][c] += 1
		tStr = tweet[0]
		if tStr.find("\"") != -1:
			XTr[i][c+1] = 1
		if tStr.find("https") != -1:
			XTr[i][c+2] = 1
		if tStr.find("#") != -1:
			XTr[i][c+3] = 1
		sid = SentimentIntensityAnalyzer()
		sentiments = sid.polarity_scores(tStr)
		XTr[i][c+4] = round(sentiments['pos'])
		XTr[i][c+5] = round(sentiments['neg'])
		XTr[i][c+6] = round(sentiments['neu'])
		s = tweet[2]
		hr = ((s.split(' '))[1].split(':'))[0]
		if hr in range(0,10):
			XTr[i][c+7] = 1
		elif hr in range(10,17):
			XTr[i][c+8] = 1
		elif hr in range(17,21):
			XTr[i][c+9] = 1
		else:
			XTr[i][c+10] = 1



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
			words = nltk.word_tokenize(text)
			idList.append(id)
			testTweetDict[id] = [text, favCount, created, retCount, words]

	XTe = np.zeros((len(testTweetDict), c+11))
	for i,(tweetId,tweet) in enumerate(testTweetDict.items()):
		tWords = tweet[-1]
		for tw in tWords:
			if tw.lower() in wordDict:
				XTe[i][wordDict[tw.lower()]] += 1
			else:
				XTe[i][c] += 1
		tStr = tweet[0]
		if tStr.find("\"") != -1:
			XTe[i][c+1] = 1
		if tStr.find("https") != -1:
			XTe[i][c+2] = 1
		if tStr.find("#") != -1:
			XTe[i][c+3] = 1
		sid = SentimentIntensityAnalyzer()
		sentiments = sid.polarity_scores(tStr)
		XTe[i][c+4] = round(sentiments['pos'])
		XTe[i][c+5] = round(sentiments['neg'])
		XTe[i][c+6] = round(sentiments['neu'])
		s = tweet[2]
		hr = ((s.split(' '))[1].split(':'))[0]
		if hr in range(0,10):
			XTe[i][c+7] = 1
		elif hr in range(10,17):
			XTe[i][c+8] = 1
		elif hr in range(17,21):
			XTe[i][c+9] = 1
		else:
			XTe[i][c+10] = 1


	# print(XTe[0])
	# fiftyAvg = 0
	# for i in range(100):
	# 	indices = np.array(range(len(tweetDict)))
	# 	np.random.shuffle(indices)
	# 	XTr = XTr[indices]
	# 	YTr = YTr[indices]
	# 	XTr80 = XTr[:int(0.8*len(tweetDict))]
	# 	YTr80 = YTr[:int(0.8*len(tweetDict))]
	# 	XTr20 = XTr[int(0.8*len(tweetDict)):]
	# 	YTr20 = YTr[int(0.8*len(tweetDict)):]

	#  clf = MultinomialNB()
	#  clf.fit(XTr80, YTr80)

	#  clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(XTr80, YTr80)

	#  preds = clf.predict(XTr20)
	#
	#
	# 	print(preds)
	# 	temp = (np.equal(preds,YTr20))
	# 	fiftyAvg += (np.sum(temp)/temp.shape)
	#   fiftyAvg/=100
	#
	# print(fiftyAvg)


	# clf = MultinomialNB()
	# clf.fit(XTr, YTr)

	# clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(XTr, YTr)

	# preds = clf.predict(XTe)

	# counter = 0
	# for i in preds:
	# 	if i==-1:
	# 		counter+=1
	#
	# print(counter)

	#print(preds)
	#temp = (np.equal(preds,YTr20))
	#print(np.sum(temp)/temp.shape)

	# with open('outputMark4.csv', 'w') as testfile:
	# 	filewriter = csv.writer(testfile, delimiter=',')
	# 	filewriter.writerow(['ID','Label'])
	# 	for i, (id,pred) in enumerate(zip(idList,preds)):
	# 		filewriter.writerow([id,int(pred)])

if __name__ == '__main__':
	main()
