import csv

def main():

	tweetDict = {}

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
			tweetDict[id] = [text, favCount, created, retCount, label]

	for tweetId,tweet in tweetDict.items():
		print(tweetId + ": " + tweet[0])



if __name__ == '__main__':
	main()