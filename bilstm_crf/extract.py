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
LABEL_END = 'I-LBL'

data = pandas.read_csv("data/tiny_label.csv", delimiter="	", index_col="jd.jd_id")
# remove duplicate index
data = data.loc[~data.index.duplicated(keep='first')]
data = data[['desc_clean', 'labels']]


def valid_label(l):
    if len(l) <= 1:
        # print('invalid label removed:' + l)
        return False
    return True


data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

print('Total # of samples: {}'.format(len(data)))

# limit dataset
data2 = data.head(1000).copy(deep=True)


# data2 = data

def desc_clean_clean(s):
    s = re.sub(r'[0-9一二三四五六七八九][.、]', ' ', s)
    s = re.sub(r'[-]{3,}|[*]{3,}', '', s)
    s = re.sub(r' ', '', s)
    return s


data2['desc_clean'] = data2['desc_clean'].map(desc_clean_clean)

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


# X_train = []
# y_train = []
# X_test = np.array(data2['desc_clean'])
# y_test = np.array(full_marks)

def get_my_data():
    return X_train, X_test, y_train, y_test


sentence_break = list('。；！.;!')


def export_my_data(every_sentence=True):
    def write_to_file(fileloc, z):
        with open(fileloc, 'w+', encoding='utf-8') as f:
            prev_is_new_line = False
            for sentence, labels in z:
                for c, l in zip(sentence, labels):
                    if c.strip() != '':
                        sentence_is_ending = every_sentence and c in sentence_break
                        if prev_is_new_line and sentence_is_ending:
                            prev_is_new_line = False
                        else:
                            f.write(c + '\t' + l + '\n')
                            if sentence_is_ending:
                                f.write('\n')
                                prev_is_new_line = True
                            else:
                                prev_is_new_line = False
                if not every_sentence:
                    f.write('\n')

    write_to_file('bilstm_crf/data_dir/train_data', zip(X_train, y_train))
    write_to_file('bilstm_crf/data_dir/test_data', zip(X_test, y_test))


if __name__ == '__main__':
    # pass
    export_my_data()
