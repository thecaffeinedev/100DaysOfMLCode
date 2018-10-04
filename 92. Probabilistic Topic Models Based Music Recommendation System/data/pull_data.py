import requests
import os
import pdb
import json

upload = "http://devapi.gracenote.com/timeline/api/1.0/audio/extract/"
download = "http://devapi.gracenote.com/timeline/api/1.0/audio/features/"

file_list = os.listdir(os.getcwd()+'/Music')
id_list = {}
#print (filelist)
counter = 0
for idx, file in enumerate(file_list):
	if counter > 100:
		break
	try:
		resp = requests.post(upload, files={'audio_file':open(os.getcwd()+'/Music/'+file,'rb')})
		jresp = resp.json()
		print(jresp)
		id = str(jresp['audio_id'])
		if id not in id_list:
			id_list[id] = file
			print(str(idx+1)+' '+file)
			
	except:
		print("Exception occurred.")
	#counter += 1
if counter != 0:
	for id in id_list:
		req = requests.get(download + id)
		data = req.json()
		with open(os.getcwd()+'/json/'+id + '.json', 'w') as outfile:
			json.dump(data, outfile)
			outfile.close()

wr = open("playlist2.txt", 'w')
for idx, file in enumerate(id_list):
	wr.write(str(idx+1)+','+file + ','+id_list[file]+'\n')
wr.close()
#features = json.loads(data['features'])
#print (features['meta'])

# Read json file
# with open('data.json', 'r') as data_file:
	# data = json.load(data_file)
# features = json.loads(data['features'])
# print (features['timeline'])



