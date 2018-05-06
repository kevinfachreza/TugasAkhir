"""
CREATED BY KEVIN
FOR TUGAS AKHIR

HOW TO RUN :

python.exe "C:\Users\Kevin\PycharmProjects\TugasAkhir\index.py"

DESKRIPSI :
JADI INI JALANNYA ADALAH
DIA BUAT TREE
NANTI HASIL TREE NYA LANGSUNG DI INSERT KE DB
JADI NYALAIN DULU XAMPP NYA

TRUS UNTUK DATA CSV NYA
DI DECLARE DEFAULT SIH DI GEJALA.CSV

"""


from anytree import Node, RenderTree, AsciiStyle, findall, find, findall_by_attr
from collections import Counter
import sys
from collections import defaultdict, namedtuple
from itertools import combinations
import mysql.connector

cnx = mysql.connector.connect(user='root', database='aqeela_tugas_akhir')
cursor = cnx.cursor()

class FrequentPattern(object):
    def __init__(self, suffix=None, pattern=None, support=None):
        self.suffix = suffix
        self.pattern = pattern
        self.support = support


def init(min_support, min_probability):
	min_support = 3;
	min_probability = 0.5;

def readFile():
	lines = [line.rstrip('\n') for line in open('C:\\Users\\Kevin\\PycharmProjects\\AqeelaTugasAkhir\\gejala.csv')]

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
			del item_counter[item]

	#print item_counter

	#declare urutan
	#kalo berdasar cuma counter, dia bakal ke mix yang mana yg duluan jika counternya sama

	itemOrder = {}

	lengthItems = len(item_counter)
	for i in range(1,lengthItems+1):
		itemOrder[item_counter.most_common()[-i][0]] = i;

	#mengurutkan masing masing array pada data sesuai dari frekuensi
	count = 0;
	for item in items:
		#filter berdasarkan threshold
		item = filter(lambda v: v in itemOrder, item)
	   	#mengurutkan item
		item.sort(key=lambda v: itemOrder[v], reverse=True)
		items[count] = item
		print str(count) + ' >>> ' + ','.join(item)
		count+=1
		


	data = [items,item_counter]
	return data

def buildTree(dataset):
	
	root = Node("root",support = 0)

	#print dataset
	#print '-----------BEGIN-----------'
	#print ''
	for data in dataset:
		parent = 0;
		for item in data:
			if parent==0:
				parent = root

			child = find(parent, filter_=lambda node: node.name == item, stop=None, maxlevel = 2)
			"""
			print 'Item : ' + item
			print 'parent'
			print parent
			print 'Child :'
			print child
			"""
			if child:
				#print "action : increment"
				child.support = child.support + 1
				parent = child 
			else:
				#print "action : add"
				new_node = Node(item, parent=parent, support = 1)
				parent = new_node


			#print ''
			#print ''


	

	return root
	
def mineFrequentPattern(dataset, tree, min_support,itemCounter):
	root = tree
	print dataset
	
	#MENGURUTKAN ITEM DARI PALING DIKIT KARENA AKAN DI BOTTOM UP, INTINYA NGEDAPETIN NODE TERBAWAH
	print itemCounter
	itemReverse = []
	lengthItems = len(itemCounter)
	for i in range(1,lengthItems+1):
		itemReverse.append(itemCounter.most_common()[-i][0])
	

	#TRACE DARI ITEM PALING BAWAH

	#DECLARE DICTIONARIES
	condPattern = {}
	frequentPattern = []

	for item in itemReverse:
		#MENCARI SEMUA NODE YANG BERATRIBUT ITEM TERSEBUT
		nodes = findall_by_attr(root, item)
		#print item
		#UNTUK SETIAP NODE
		#DICARI PARENT
		#PARENT AKAN DISIMPAN DI DALAM ARRAY

		condPatternParents = []
		for node in nodes:
			parent = str(node)
			parent = parent.replace("/root/","")
			parent = parent.split("'")[1]
			for i in range(0, node.support):
				condPatternParents.append(parent)


		condPattern[item] = condPatternParents


	#print condPattern
	for item in itemReverse:
		print ''
		print str(item) + " >>>> "+ str(condPattern[item])

		#MEMASUKKAN SEMUA ITEM UNTUK DIHITUNG
		itemsInPattern = []
		for pattern in condPattern[item]:
			#print pattern
			temp_pattern = pattern.split('/')
			for temp_item in temp_pattern:
				itemsInPattern.append(temp_item)

		print itemsInPattern
		item_counter = Counter(itemsInPattern)
		print item_counter

		temp_item_counter = Counter(item_counter)

		#dia gabisa di delete waktu di iterasi, jadi ya iterasi data copyannya baru di delete
		#di delete agar array yang ga masuk threshold langsung ilang, nanti diilangin pake lamda
		for temp_item in temp_item_counter:
			if(temp_item_counter[temp_item] < min_support):
				del item_counter[temp_item]

		print item_counter
		willBeCombinatedItem = []
		willBeCombinatedItem = filter(lambda v: v in item_counter, itemReverse)
		print willBeCombinatedItem
		print ''
		for L in range(1, len(willBeCombinatedItem)+1):
  			for resultcombo in combinations(willBeCombinatedItem, L):
  				resultcombo = list(resultcombo)
  				if item in resultcombo:
  					#print resultcombo
  					#setiap kombo yang terbuat
  					#harus di cek, berapa kali dia muncul diantara condition pattern yang telah ada
  					
  					support = 0
  					for pattern in condPattern[item]:
						pattern_array = pattern.split('/')
						flag_support_increment = 1

						for combo in resultcombo:
							if combo not in pattern_array:
								flag_support_increment = 0

  							
						if flag_support_increment == 1:
							support += 1


					if support >= min_support:
						print ','.join(resultcombo) + ' >>> ' + str(support)
  						frequentPattern.append(FrequentPattern(item,','.join(resultcombo),str(support)))


  					"""
  					flag = 0
  					#untuk setiap item pada array combo
  					#akan dicari minimal support yang ada
  					for combo in resultcombo:
  						current_support = item_counter[combo]
  						if flag == 0:
  							support = current_support
  							flag = 1
  						elif(current_support < support):
  							support = current_support

  					SALAH KARENA HARUSNYA GA NGITUNG SINGLE PER ITEM AJA

					"""
  					
		print ''
	print ''

	for fp in frequentPattern:
		print fp.suffix + ' -- ' + fp.pattern + ' >>> ' + fp.support 

	return frequentPattern

def find_by_item(list, keyword):

    #setiap datalist, akan di iterasi
    #jika itemnya kaya di keyword maka return support
    #item.item = A,B,C bentuknya string bukan array

    for item in list:
        if item.pattern == keyword:
            return item.support

    return -1


def generate_rules(filteredList,threshold):
    print ""
    print "----RULES----"
    print ""


    for i in range(len(filteredList)):
        subset = filteredList[i].pattern.split(',');
        #print subset
        for L in range(1, len(subset)):
            for resultsubset in combinations(subset, L):
                #convert tuple into list array
                left = list(resultsubset)
                right = list(subset)

                #menghilangkan item yang ada di kanan berdasarkan yang dikir
                #subset = A,B,C left = A,C right = A,B,C
                #subset = A,B,C left = A,C right = B
                for item in left:
                    right.remove(item)

                #array jadi string
                left_string = ','.join(left)
                right_string = ','.join(right)
                subset_string =  ','.join(subset)

                #hitung support
                support_subset = find_by_item(filteredList, subset_string)
                support_left = find_by_item(filteredList, left_string)
                support = float(float(support_subset)/float(support_left))

                #disort, biar kalo ke query tidak terjadi 5,10 dan 10,5. Nah diurutin jadi pasti yang kecil di depan
                


                right = sorted(right, key=lambda x: int(x))
                left = sorted(left, key=lambda x: int(x))

                left_string = ','.join(left)
                right_string = ','.join(right)

                if(support < 0):
                	print 'BUGMIN'

                #menghilangkan spasi
                left_string = left_string.replace(" ","")
                right_string = right_string.replace(" ","")

                #print 'RULES :'+ str(subset_string) + " || " + str(left_string) + " >> " + str(right_string) + " || " + str(support_subset) + " / " + str(support_left) + " = " + str(support)
                query = 'INSERT INTO rules(items,result,probability) VALUES("'+left_string+'","'+right_string+'","'+str(support)+'");'
                print query
                cursor.execute(query)
                cnx.commit()


        #print ""


if __name__ == '__main__':


	database =[]
	min_support = 5;
	min_probability = 0.5;


	#init(min_support, min_probability)

	database = readFile()
	itemDataset = cleanData(database,min_support)
	itemOrdered = itemDataset[0]
	itemCounter = itemDataset[1]
	#print itemOrdered

	tree = buildTree(itemOrdered)
	fp = mineFrequentPattern(itemOrdered, tree, min_support, itemCounter)
	rules = generate_rules(fp, min_support)
	#TREE
	#print ''
	#print '------------------------TREE------------------------'
	#print ''
	#for pre, fill, node in RenderTree(tree):
	#	print("%s%s, support %d" % (pre, node.name,node.support))


	cursor.close()
	cnx.close()