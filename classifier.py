import csv
import nltk
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC

def readData(fName, isTraining):
	tweetDict = {}
	wordDict = {}
	c = 0
	idList = []

	with open(fName,'r') as tFile:
		tReader = csv.reader(tFile, delimiter = '\n')
		for i,tweet in enumerate(tReader):
			if not tweet:
				break
			if i == 0:
				continue
			tweet = tweet[0].strip('[').strip(']').split(',')
			id, text, favCount, created, retCount = tweet[0],tweet[1],tweet[3],tweet[5],tweet[12]
			if isTraining:
				label = tweet[17]
			words = nltk.word_tokenize(text)

			if isTraining:
				for w in words:
					if not(w.lower() in wordDict):
						wordDict[w.lower()] = c
						c += 1
				tweetDict[id] = [text, favCount, created, retCount, label, words]
			else:
				idList.append(id)
				tweetDict[id] = [text, favCount, created, retCount, words]

	if isTraining:
		return tweetDict,wordDict,c
	else:
		return tweetDict, idList

def createX(tweetDict, wordDict, c):

	X = np.zeros((len(tweetDict), c+11))
	for i,(tweetId,tweet) in enumerate(tweetDict.items()):
		tWords = tweet[-1]
		for tw in tWords:
			if tw.lower() in wordDict:
				X[i][wordDict[tw.lower()]] += 1
			else:
				X[i][c] += 1
		tStr = tweet[0]
		if tStr.find("\"") != -1:
			X[i][c+1] = 1
		if tStr.find("https") != -1:
			X[i][c+2] = 1
		if tStr.find("#") != -1:
			X[i][c+3] = 1
		sid = SentimentIntensityAnalyzer()
		sentiments = sid.polarity_scores(tStr)
		X[i][c+4] = round(sentiments['pos'])
		X[i][c+5] = round(sentiments['neg'])
		X[i][c+6] = round(sentiments['neu'])
		s = tweet[2]
		hr = ((s.split(' '))[1].split(':'))[0]
		if hr in range(0,10):
			X[i][c+7] = 1
		elif hr in range(10,17):
			X[i][c+8] = 1
		elif hr in range(17,21):
			X[i][c+9] = 1
		else:
			X[i][c+10] = 1

	return X

def createYTr(tweetDict):

	YTr = np.array([tweet[-2] for (tweetId,tweet) in tweetDict.items()])
	return YTr

def main():

	tweetDict,wordDict,c = readData('train.csv',True)
	XTr = createX(tweetDict, wordDict, c)
	YTr = createYTr(tweetDict)

	testTweetDict, idList = readData('test.csv',False)
	XTe = createX(testTweetDict, wordDict, c)

	# fiftyAvg = 0
	# for i in range(50):
	# 	indices = np.array(range(len(tweetDict)))
	# 	np.random.shuffle(indices)
	# 	XTr = XTr[indices]
	# 	YTr = YTr[indices]
	# 	XTr80 = XTr[:int(0.8*len(tweetDict))]
	# 	YTr80 = YTr[:int(0.8*len(tweetDict))]
	# 	XTr20 = XTr[int(0.8*len(tweetDict)):]
	# 	YTr20 = YTr[int(0.8*len(tweetDict)):]
	
	# 	#clf = LinearSVC(random_state=0, tol=1e-5)
	# 	#clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(XTr80, YTr80)
	# 	clf = MultinomialNB()
	# 	clf.fit(XTr80, YTr80)
	# 	preds = clf.predict(XTr20)
	# 	temp = (np.equal(preds,YTr20))
	# 	fiftyAvg += (np.sum(temp)/temp.shape)
	# fiftyAvg/=50
	# print(fiftyAvg)

	 # clf = MultinomialNB()
	 # clf.fit(XTr80, YTr80)
		#print(preds)


	# clf = MultinomialNB()


	#clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(XTr, YTr)
	clf = LinearSVC(random_state=0, tol=1e-5)
	clf.fit(XTr, YTr)

	preds = clf.predict(XTe)

	# counter = 0
	# for i in preds:
	# 	if i==-1:
	# 		counter+=1
	#
	# print(counter)

	#print(preds)
	#temp = (np.equal(preds,YTr20))
	#print(np.sum(temp)/temp.shape)

	with open('outputMark4.5.csv', 'w') as testfile:
		filewriter = csv.writer(testfile, delimiter=',')
		filewriter.writerow(['ID','Label'])
		for i, (id,pred) in enumerate(zip(idList,preds)):
			filewriter.writerow([id,int(pred)])

if __name__ == '__main__':
	main()
