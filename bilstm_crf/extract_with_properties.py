import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import jieba
import jieba.posseg as pseg
import pickle

from extract_util import desc_clean_clean, filter_seg_result, valid_label, write_to_file
from util import find_locations, a2g

LABEL_BEGIN = 'B-LBL'
LABEL_MIDDLE = 'I-LBL'
LABEL_END = 'I-LBL'

data = pandas.read_csv("data/tiny_label.csv", delimiter="	", index_col="jd.jd_id")
# remove duplicate index
data = data.loc[~data.index.duplicated(keep='first')]
data = data[['desc_clean', 'labels']]

data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

labels_set = set()
for label_list in data['labels']:
    for label in label_list:
        labels_set.add(label)
for label in labels_set:
    jieba.add_word(label)

print('Total # of samples: {}'.format(len(data)))

# limit dataset
# data2 = data.head(1000).copy(deep=True)
data2 = data

data2['desc_clean'] = data2['desc_clean'].map(desc_clean_clean)

full_marks = []

# x, y = list(zip(data2['desc_clean'], data2['labels']))[0]

marks_set = {
    LABEL_BEGIN, LABEL_MIDDLE, LABEL_END
}


processed_desc = []

for x, y in zip(data2['desc_clean'], data2['labels']):
    seg_result = list(filter(filter_seg_result, pseg.cut(x)))

    x = ''.join([w for w, p in seg_result])
    processed_desc.append(x)

    full_marks.append(['O'] * len(x))
    y.sort(key=lambda l: len(l), reverse=True)
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

    seg_result = a2g(seg_result)

    seg_word = ''
    properti = ''
    begin = True
    skip_counter = 0
    for i, ch in enumerate(x):
        if seg_word == '':
            seg_word, properti = next(seg_result)
            begin = True
        if begin:
            mark_prefix = 'B-'
            begin = False
        else:
            mark_prefix = 'I-'
        if seg_word[0] == ch:
            skip_counter = 0
            mark = mark_prefix + properti.upper()
            if full_marks[-1][i] == 'O':
                full_marks[-1][i] = mark
                marks_set.add(mark)
            seg_word = seg_word[1:]
        else:
            skip_counter += 1
        if skip_counter >= 5:
            print(ch)

X_train, X_test, y_train, y_test = train_test_split(np.array(processed_desc), np.array(full_marks), test_size=0.15)


def get_my_data():
    return X_train, X_test, y_train, y_test


sentence_break = list('。；！.;!')


def export_my_data():
    train_results = write_to_file('bilstm_crf/data_dir/train_data', zip(X_train, y_train))
    test_results = write_to_file('bilstm_crf/data_dir/test_data', zip(X_test, y_test))
    return train_results, test_results


if __name__ == '__main__':
    # pass
    train_results, test_results = export_my_data()

    tag2label = {
        'O': 0
    }
    for i, lbl in enumerate(marks_set):
        tag2label[lbl] = i + 1
    print(tag2label)
    with open('bilstm_crf/data_dir/tag2label.pkl', 'wb+') as f:
        pickle.dump(tag2label, f)
