import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import jieba.posseg as pseg
import pickle

from extract_util import desc_clean_clean, filter_seg_result, valid_label, write_to_file, simplify_property
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

print('Total # of samples: {}'.format(len(data)))

# limit dataset
# data2 = data.head(1000).copy(deep=True)
data2 = data

data2['desc_clean'] = data2['desc_clean'].map(desc_clean_clean)

full_marks = []

# x, y = list(zip(data2['desc_clean'], data2['labels']))[0]

prop_set = set()

processed_desc_w_prop = []

for x, y in zip(data2['desc_clean'], data2['labels']):
    seg_result = list(filter(filter_seg_result, pseg.cut(x)))

    x = ''.join([w for w, p in seg_result])

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
    prop_labels = []

    seg_word = ''
    properti = ''
    begin = True
    for ch in x:
        if seg_word == '':
            seg_word, properti = next(seg_result)
            # simplify

            properti = simplify_property(properti)

            begin = True
        if begin:
            mark_prefix = 'B-'
            begin = False
        else:
            mark_prefix = 'I-'
        mark = mark_prefix + properti.upper()
        prop_labels.append(mark)
        prop_set.add(mark)
        seg_word = seg_word[1:]

    processed_desc_w_prop.append((x, prop_labels))

X_train, X_test, y_train, y_test = train_test_split(np.array(processed_desc_w_prop), np.array(full_marks), test_size=0.15)


def get_my_data():
    return X_train, X_test, y_train, y_test


sentence_break = list('。；！.;!')


def export_my_data():

    def write_to_file_w_prop(fileloc, z, every_sentence=True):
        sentences_record = []
        propss_record = []
        with open(fileloc, 'w+', encoding='utf-8') as f:
            prev_is_new_line = False
            for (sentence, props), labels in z:
                sentence_record = ''
                props_record = []
                for (c, p), l in zip(zip(sentence, props), labels):
                    if c.strip() != '':
                        sentence_is_ending = every_sentence and c in sentence_break
                        if prev_is_new_line and sentence_is_ending:
                            prev_is_new_line = False
                        else:
                            f.write(c + ' ' + p + ' ' + l + '\n')
                            sentence_record += c
                            props_record.append(p)
                            if sentence_is_ending:
                                f.write('\n')
                                sentences_record.append(sentence_record)
                                sentence_record = ''
                                propss_record.append(props_record)
                                props_record = []
                                prev_is_new_line = True
                            else:
                                prev_is_new_line = False
                if not every_sentence:
                    f.write('\n')
        return sentences_record, propss_record

    train_results = write_to_file_w_prop('bilstm_crf/data_dir/train_data', zip(X_train, y_train))
    test_results = write_to_file_w_prop('bilstm_crf/data_dir/test_data', zip(X_test, y_test))
    return train_results, test_results


if __name__ == '__main__':
    # pass
    train_results, test_results = export_my_data()

    prop2label = {
        '<UNK>': 0
    }
    for i, lbl in enumerate(prop_set):
        prop2label[lbl] = i + 1
    print(prop2label)
    with open('bilstm_crf/data_dir/prop2label.pkl', 'wb+') as f:
        pickle.dump(prop2label, f)
