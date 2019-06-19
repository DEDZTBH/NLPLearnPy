import pandas
import numpy as np
from sklearn.model_selection import train_test_split

from extraction.util import find_locations, random_rows

LABEL_BEGIN = 'B-LBL'
LABEL_MIDDLE = 'I-LBL'

data = pandas.read_csv("data/tiny_label.csv", delimiter="	")
data = data[['desc_clean', 'labels']]


def valid_label(l):
    if len(l) <= 1:
        # print('invalid label removed:' + l)
        return False
    return True


data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

full_marks = []

for x, y in zip(data['desc_clean'], data['labels']):
    full_marks.append(['O'] * len(x))
    y.sort(key=lambda x: len(x), reverse=True)
    for label in y:
        found_locs = list(find_locations(x, label))
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

X_train, X_test, y_train, y_test = train_test_split(data['desc_clean'], full_marks, test_size=0.15)

with open('extraction/work/data_dir/train.txt', 'w') as f:
    for sentence, labels in zip(X_train, y_train):
        for c, l in zip(sentence, labels):
            f.write(c + ' ' + l + '\n')
        f.write('\n')

with open('extraction/work/data_dir/dev.txt', 'w') as f:
    for sentence, labels in zip(X_test, y_test):
        for c, l in zip(sentence, labels):
            f.write(c + ' ' + l + '\n')
        f.write('\n')

with open('extraction/work/data_dir/test.txt', 'w') as f:
    for sentence, labels in random_rows(np.array([data['desc_clean'], full_marks]).transpose(), 100):
        for c, l in zip(sentence, labels):
            f.write(c + ' ' + l + '\n')
        f.write('\n')
