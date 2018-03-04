from anytree import Node, RenderTree, AsciiStyle
from collections import Counter
import sys
from collections import defaultdict, namedtuple

def init(min_support, min_probability):
	min_support = 4;
	min_probability = 0.5;

def readFile():
	lines = [line.rstrip('\n') for line in open('C:\\Users\\Kevin\\PycharmProjects\\AqeelaTugasAkhir\\testing.csv')]

	gejala = []
	for val in lines:
	    detail_gejala = val.split(',')
	    for val2 in detail_gejala:
	        val2 = val2
	    gejala.append(detail_gejala)

	return gejala;

def cleanData(data,min_support):
	items = data
	
	#ngitung frekuensi masing masing item
	item_counter = Counter(x for sublist in data for x in sublist)
	#copy untuk iterasi
	temp_item_counter = Counter(item_counter)


	#dia gabisa di delete waktu di iterasi, jadi ya iterasi data copyannya baru di delete
	#di delete agar array yang ga masuk threshold langsung ilang, nanti diilangin pake lamda
	for item in temp_item_counter:
		if(temp_item_counter[item] < min_support):
			del item_counter['e']



	#mengurutkan masing masing array pada data sesuai dari frekuensi
	count = 0;
	for item in items:
		#filter berdasarkan threshold
		item = filter(lambda v: v in item_counter, item)
	   	#mengurutkan item
		item.sort(key=lambda v: item_counter[v], reverse=True)
		items[count] = item
		count+=1
	return items

def buildTree(dataset):
	
	root = Node("root")


	for data in dataset:
		parent = 0;
		for item in data:
			if parent==0:
				parent = root

			child = findall(parent, filter_=lambda node: node.name in (item))
			findall(parent, filter_=lambda node: node.name in ("a", "b"))
			print child

			parent = Node(item, parent=parent, support = 1)




	for pre, fill, node in RenderTree(root):
		print("%s%s" % (pre, node.name))

	

if __name__ == '__main__':


	database =[]
	min_support = 3;
	min_probability = 0.5;


	#init(min_support, min_probability)

	database = readFile()
	itemOrdered = cleanData(database,min_support)
	#print itemOrdered

	tree = buildTree(itemOrdered)


	