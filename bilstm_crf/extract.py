import pandas
from sklearn.model_selection import train_test_split
import numpy as np

from extract_util import desc_clean_clean, valid_label, write_to_file
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

data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

print('Total # of samples: {}'.format(len(data)))

# limit dataset
# data2 = data.head(1000).copy(deep=True)
data2 = data

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


def export_my_data():
    train_results = write_to_file('bilstm_crf/data_dir/train_data', zip(X_train, y_train))
    test_results = write_to_file('bilstm_crf/data_dir/test_data', zip(X_test, y_test))
    return train_results, test_results


if __name__ == '__main__':
    # pass
    train_results, test_results = export_my_data()
