import os 

dataset = 'office-home'

if dataset == 'office':
	domains = ['amazon', 'dslr', 'webcam']
elif dataset == 'office-caltech':
	domains = ['amazon', 'dslr', 'webcam', 'caltech']
elif dataset == 'office-home':
	domains = ['Art', 'Clipart', 'Product', 'Real_World']
else:
	print('No such dataset exists!')

for domain in domains:
	log = open('./data/'+dataset+'/'+domain+'_list.txt','w')
	directory = os.path.join('./data', dataset, os.path.join(domain,'images'))
	print(directory)
	classes = [x[0] for x in os.walk(directory)]
	print(classes)
	classes = classes[1:]
	classes.sort()
	for idx,f in enumerate(classes):
		files = os.listdir(f)
		for file in files:
			s = os.path.abspath(os.path.join(f,file)) + ' ' + str(idx) + '\n'
			log.write(s)
	log.close()