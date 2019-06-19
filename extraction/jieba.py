import pandas
import jieba.analyse

jieba.initialize()

with open('data/stopwords.txt', 'r') as file:
    stop_words = file.read().split('\n')

data = pandas.read_csv("data/tiny_label.csv", delimiter="	")
data = data[['desc_clean', 'labels']]
data['labels'] = [list(filter(lambda x: x != '', eval(l))) for l in data['labels']]

for x, y in zip(data['desc_clean'], data['labels']):
    seglist = list(filter(lambda w: w not in stop_words, jieba.cut(x)))
    miss_list = list(filter(lambda label: label not in seglist, y))
    if len(miss_list) > 0:
        print("missing:" + str(miss_list))
