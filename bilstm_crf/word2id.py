import pickle


def word2id():
    with open('data/vocab.txt', 'r', encoding='utf-8') as f:
        vocabs = f.read().split('\n')
    word2id_dict = {}
    for i, w in enumerate(vocabs):
        word2id_dict[w] = i + 1

    with open('data/word2id.pkl', 'wb') as f:
        pickle.dump(word2id_dict, f)


if __name__ == '__main__':
    word2id()
