import pyfpgrowth
import numpy


transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]

transactions2 = [["cola", "bread", "soap"],
                ["bread", "candy"],
                ["bread", "noodle"],
                ["cola", "bread", "candy"],
                ["cola", "noodle"],
                ["bread", "noodle"],
                ["cola", "noodle"],
                ["cola", "bread", "noodle", "soap"],
                ["cola", "bread", "noodle"]]

transactions3 = [["M", "O", "N","K","E","Y"],
                ["D","O","N","K","E","Y"],
                ["M", "A","K","E"],
                ["M", "U", "C","K","Y"],
                ["C", "O","O","K","I","E"]]

gejala2 = [[1, 2, 3, 4 ],
           [5, 6, 7, 8, 9, 10, 11, 12, 30 ],
           [6, 13, 14, 9, 48, 49 ],
           [5, 15, 9, 16, 17, 18, 19, 6 ],
           [6, 2, 20, 21 ],
           [23, 24, 25, 26, 27, 28, 29 ],
           [24, 31, 23, 5, 10, 27, 32, 33, 28, 34, 35, 11, 37, 36, 38, 39, 40 ],
           [24, 31, 52, 23, 41, 27, 28, 34, 45, 43, 44, 35, 11, 38, 47, 46 ]
           ]


lines = [line.rstrip('\n') for line in open('C:\\Users\\Kevin\\PycharmProjects\\AqeelaTugasAkhir\\testing.csv')]


gejala = []
for val in lines:
    detail_gejala = val.split(',')
    print detail_gejala
    for val2 in detail_gejala:
        val2 = val2
    gejala.append(detail_gejala)



patterns = pyfpgrowth.find_frequent_patterns(gejala, 2)
rules = pyfpgrowth.generate_association_rules(patterns, 0)
#print(patterns)

new_rules = rules.split(', (');

for rule in new_rules:
  print rule


