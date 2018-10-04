import requests
import os
import json
import re

download = "http://devapi.gracenote.com/timeline/api/1.0/audio/features/"
# download json files from server
with open('playlist.txt', 'r') as datafile:
	id_list = [i.split(',')[1] for i in datafile.readlines()]
#print(id_list)
for id in id_list:
		print("Downloading json file from server: " + id)
		req = requests.get(download + id)
		data = req.json()
		with open(os.getcwd()+'/json/'+id + '.json', 'w') as outfile:
			json.dump(data, outfile)
			outfile.close()

file_list = os.listdir(os.getcwd()+'/json')
counter = 0
wr = open('mood.txt', 'w')
for filename in file_list:
	print(filename)
	with open(os.getcwd()+'/json/'+filename, 'r') as data_file:
		data = json.load(data_file)
		data_file.close()
	mood = [filename[:-5]] # stores the moods for each song
	features = json.loads(data['features'])

	for m in features['timeline']['mood']: # each interval
		val = m['values']
		for d in val: # each mood in the interval
			if d['score'] >= 10:
				mood += filter(None, re.split(r'/+| +|&+', d['label']))
	counter += 1
	print(counter, len(mood))
	wr.write(','.join(mood)+'\n')
wr.close()