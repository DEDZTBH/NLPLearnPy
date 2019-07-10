import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import re

from extract_util import valid_label
from util import find_locations, random_rows

with open('data/stopwords2.txt', 'r') as file:
    stop_words = file.read().split('\n')
stop_words.sort(key=lambda x: len(x), reverse=True)

LABEL_BEGIN = 'B-LBL'
LABEL_MIDDLE = 'I-LBL'
with open('bert_bilstm_crf_ner/work/data_dir/labels.txt', 'w') as f:
    f.write(LABEL_BEGIN)
    f.write("\n")
    f.write(LABEL_MIDDLE)

data = pandas.read_csv("data/tiny_label.csv", delimiter="	")
data = data[['desc_clean', 'labels']]

data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

# limit dataset
data = data.head(256)

for sw in stop_words:
    for i, str in enumerate(data['desc_clean']):
        data['desc_clean'][i] = re.sub(r'[-]{3,}|[*]{3,}', '', str.replace(sw, ''))

full_marks = []

for x, y in zip(data['desc_clean'], data['labels']):
    full_marks.append(['O'] * len(x))
    y.sort(key=lambda x: len(x), reverse=True)
    for label in y:
        found_locs = list(find_locations(x, label))
        if len(found_locs) == 0:
            print(label + ' not found in ' + x)
        for found_location in found_locs:
            if full_marks[-1][found_location] == 'O':
                full_marks[-1][found_location] = LABEL_BEGIN
            else:
                # pass
                print('-----------------------Intersecting labels @ b-----------------------')
                print(x)
                print(y)
                print(label)
                print(x[found_location])
                print(full_marks[-1][found_location])
            for i in range(found_location + 1, found_location + len(label)):
                if i < len(x):
                    if full_marks[-1][i] == 'O':
                        full_marks[-1][i] = LABEL_MIDDLE
                    else:
                        pass
                        # print('-----------------------Intersecting labels @ i-----------------------')
                        # print(x)
                        # print(y)
                        # print(label)
                        # print(x[i])
                        # print(full_marks[-1][i])

X_train, X_test, y_train, y_test = train_test_split(data['desc_clean'], full_marks, test_size=0.1)

with open('bert_bilstm_crf_ner/work/data_dir/train.txt', 'w') as f:
    for sentence, labels in zip(X_train, y_train):
        for c, l in zip(sentence, labels):
            f.write(c + ' ' + l + '\n')
        f.write('\n')

with open('bert_bilstm_crf_ner/work/data_dir/dev.txt', 'w') as f:
    for sentence, labels in zip(X_test, y_test):
        for c, l in zip(sentence, labels):
            f.write(c + ' ' + l + '\n')
        f.write('\n')

with open('bert_bilstm_crf_ner/work/data_dir/test.txt', 'w') as f:
    for sentence, labels in random_rows(np.array([data['desc_clean'], full_marks]).transpose(), 10):
        for c, l in zip(sentence, labels):
            f.write(c + ' ' + l + '\n')
        f.write('\n')
