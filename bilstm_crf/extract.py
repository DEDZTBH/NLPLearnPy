import pandas
from sklearn.model_selection import train_test_split
import re
import numpy as np

from util import find_locations, random_rows

# with open('data/stopwords2.txt', 'r', encoding='utf-8') as file:
#     stop_words = file.read().split('\n')
# stop_words.sort(key=lambda x: len(x), reverse=True)

LABEL_BEGIN = 'B-LBL'
LABEL_MIDDLE = 'I-LBL'
LABEL_END = 'E-LBL'

data = pandas.read_csv("data/tiny_label.csv", delimiter="	")
data = data[['desc_clean', 'labels']]


def valid_label(l):
    if len(l) <= 1:
        # print('invalid label removed:' + l)
        return False
    return True


data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

# limit dataset
data2 = data.head(256).copy(deep=True)

for i in range(len(data2['desc_clean'])):
    # for sw in stop_words:
    #     data['desc_clean'][i] = data['desc_clean'][i].replace(sw, '')
    # data['desc_clean'][i] = re.sub(r'[0-9一二三四五六七八九][.、]', ' ', data['desc_clean'][i])
    # data['desc_clean'][i] = re.sub(r'[0-9、]+', ' ', data['desc_clean'][i])
    data2['desc_clean'][i] = re.sub(r'[-]{3,}|[*]{3,}', '', data2['desc_clean'][i])
    # data['desc_clean'][i] = re.sub(r'【.*?】', '', data['desc_clean'][i])

full_marks = []

for x, y in zip(data2['desc_clean'], data2['labels']):
    full_marks.append(['O'] * len(x))
    y.sort(key=lambda x: len(x), reverse=True)
    for label in y:
        found_locs = list(find_locations(x, label))
        label_len = len(label)
        if len(found_locs) == 0:
            print(label + ' not found in ' + x)
        for found_location in found_locs:
            if full_marks[-1][found_location] == 'O':
                full_marks[-1][found_location] = LABEL_BEGIN
            else:
                print('-----------------------Intersecting labels @ b-----------------------')
                print(x)
                print(y)
                print(label)
                print(x[found_location - 1] + x[found_location] + x[found_location + 1])
                print(full_marks[-1][found_location])

            if label_len > 1:
                if full_marks[-1][found_location + label_len - 1] == 'O':
                    full_marks[-1][found_location + label_len - 1] = LABEL_END
                else:
                    pass

            if label_len > 2:
                for i in range(found_location + 1, found_location + label_len - 1):
                    if i < len(x):
                        if full_marks[-1][i] == 'O':
                            full_marks[-1][i] = LABEL_MIDDLE
                        else:
                            pass

X_train, X_test, y_train, y_test = train_test_split(np.array(data2['desc_clean']), np.array(full_marks), test_size=0.1)


def get_my_data():
    return X_train, X_test, y_train, y_test


sentence_break = list('。；！.;!')


def export_my_data(every_sentence=True):
    def writeToFile(fileloc, z):
        with open(fileloc, 'w+', encoding='utf-8') as f:
            for sentence, labels in z:
                for c, l in zip(sentence, labels):
                    f.write(c + '\t' + l + '\n')
                    if every_sentence and c in sentence_break:
                        f.write('\n')
                if not every_sentence:
                    f.write('\n')

    writeToFile('bilstm_crf/data_dir/train_data', zip(X_train, y_train))
    writeToFile('bilstm_crf/data_dir/test_data', zip(X_test, y_test))
    # writeToFile('bilstm_crf/data_dir/test.txt', random_rows(np.array([data2['desc_clean'], full_marks]).transpose(), 5))


if __name__ == '__main__':
    # pass
    export_my_data()
