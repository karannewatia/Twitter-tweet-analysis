import csv
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def readData(fName, isTraining):
	tweetDict = {}
	wordDict = {}
	tupDict = {}
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

			#pos_tags = nltk.pos_tag(words)


			if isTraining:
				for w in words:
					if not(w.lower() in wordDict):
						wordDict[w.lower()] = c
						c += 1
				# for tup in pos_tags:
				# 	if not (tup in tupDict):
				# 		tupDict[tup] = c
				# 		c +=1

				tweetDict[id] = [text, favCount, created, retCount, label, words]
				# tweetDict[id] = [text, favCount, created, retCount, label, pos_tags]
			else:
				idList.append(id)
				tweetDict[id] = [text, favCount, created, retCount, words]
				# tweetDict[id] = [text, favCount, created, retCount, pos_tags]

	if isTraining:
		# return tweetDict,wordDict,c
		return tweetDict, tupDict, c
	else:
		return tweetDict, idList

def createX(tweetDict, wordDict, c):

	X = np.zeros((len(tweetDict), c+11+35))
	for i,(tweetId,tweet) in enumerate(tweetDict.items()):
		tWords = tweet[-1]
		for tw in tWords:
			if tw.lower() in wordDict:
			#if tw in wordDict:
				X[i][wordDict[tw.lower()]] += 1
				#X[i][wordDict[tw]] += 1

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

		pos_tags = nltk.pos_tag(tWords)
		categoryDict = {}
		categories = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
		for cat in categories:
			categoryDict[cat] = 0
		for word, cat in pos_tags:
				if cat in categories:
					categoryDict[cat] += 1

		# tempCounter = c+10
		# for cat, count in categoryDict.items():
		# 	tempCounter += 1
		# 	X[i][tempCounter] = count

		# totNumOfFeatures = tempCounter+1

		# newX = np.zeros((len(tweetDict), totNumOfFeatures))

		# for vector in range((len(tweetDict))):
		# 	newX[vector] = X[vector,:totNumOfFeatures]

	return X

def createYTr(tweetDict):

	YTr = np.array([tweet[-2] for (tweetId,tweet) in tweetDict.items()])
	return YTr

def main():
	# Change this boolean when you're alternating between testing 80 20 and generating a full submission
	generatingSubmission = False

	tweetDict,wordDict,c = readData('train.csv',True)
	XTr = createX(tweetDict, wordDict, c)
	YTr = createYTr(tweetDict)

	testTweetDict, idList = readData('test.csv',False)
	XTe = createX(testTweetDict, wordDict, c)

	if not generatingSubmission:
		fiftyAvg = 0
		for i in range(100):
			indices = np.array(range(len(tweetDict)))
			np.random.shuffle(indices)
			XTr = XTr[indices]
			YTr = YTr[indices]
			XTr80 = XTr[:int(0.8*len(tweetDict))]
			YTr80 = YTr[:int(0.8*len(tweetDict))]
			XTr20 = XTr[int(0.8*len(tweetDict)):]
			YTr20 = YTr[int(0.8*len(tweetDict)):]
			#clf = LinearSVC(random_state=0, tol=1e-5)
			#clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(XTr80, YTr80)
			rowDim, columnDim = XTr80.shape

			clf = RandomForestClassifier(n_estimators=int(np.sqrt(columnDim)), max_depth=100, random_state=0)
			clf.fit(XTr80, YTr80)
			# model = SelectFromModel(clf, threshold=1e-4, prefit=True)
			# XTr80 = model.transform(XTr80)
			# idx = model.get_support(True)
			# XTr20 = XTr20[:,idx]
			# clf = RandomForestClassifier(n_estimators=int(np.sqrt(len(XTr80[0]))), max_depth=100, random_state=0)
			# clf.fit(XTr80, YTr80)
			preds = clf.predict(XTr20)
			#temp = np.equal(preds,YTr20)
			#fiftyAvg += (np.sum(temp)/preds.shape)
			fiftyAvg += (np.sum(preds==YTr20)/preds.shape)
		fiftyAvg/=100
		print(fiftyAvg)

	else:
		#clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(XTr, YTr)
		# #clf = LinearSVC(random_state=0, tol=1e-5)
		clf = RandomForestClassifier(n_estimators=int(np.sqrt(len(XTr[0]))), max_depth=100, random_state=0)

		clf.fit(XTr, YTr)
		preds = clf.predict(XTe)

		with open('outputMark16.csv', 'w') as testfile:
			filewriter = csv.writer(testfile, delimiter=',')
			filewriter.writerow(['ID','Label'])
			for i, (id,pred) in enumerate(zip(idList,preds)):
				filewriter.writerow([id,int(pred)])

	# counter = 0
	# for i in preds:
	# 	if i=='-1':
	# 		counter+=1
	# # print(preds)
	#
	# print(counter)

	#print(preds)
	#temp = (np.equal(preds,YTr20))
	#print(np.sum(temp)/temp.shape)

if __name__ == '__main__':
	main()
