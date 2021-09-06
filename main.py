# -*- coding: utf-8 -*-
#!/usr/local/bin/python3
import time
import os
import MeCab
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    start = time.time()

    path = "./"
    files = os.listdir(path)
    files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
    files_file.remove("main.py")
    print(files_file)


    #key: x, value:[w1,w2,...,wn]
    dicfile = {}

    # 文のファイルを個別に読み込む
    for file in files_file:
        with open(file,mode='r',encoding='utf-8') as f:
            # ファイル全体をテキストで読み込み
            text = f.read()
        f.close()

        # 文を形態素解析する
        tagger = MeCab.Tagger()
        parseline = tagger.parse(text).splitlines()


        thesis = []
        for line in parseline:
            #line = ['本稿\t名詞,一般,*,*,*,*,本稿,ホンコウ,ホンコー', ...
            w = line.split('\t')
            if(w[0] == "EOS"):
                break
            #w[0] = 本稿
            thesis.append(w[0])
        dicfile[file] = thesis

    # tfidfvectorizerの入力へ変換
    corpus = []
    filename = []
    for k,v in dicfile.items():
        corpus.append(' '.join(v))
        filename.append(k)

    print(corpus)

    # TF-IDF化する
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print('shape: {}'.format(X.shape))
    #shape: (3, 448)

    tfidf_X = X.toarray()

    print(tfidf_X)
    #[[0.01634868 0.         0.03269737 ... 0.         0.02768073 0.        ]
    # [0.01353335 0.02291396 0.01353335 ... 0.         0.         0.        ]
    # [0.07889686 0.         0.01972421 ... 0.033396   0.         0.16697999]]

    # 各文書のTF-IDF値に基づいて降順ソートされたindexを取得
    index = tfidf_X.argsort(axis=1)[:,::-1]
    print(index)
    #[[214  19  25 ... 315 316 447]
    # [ 65 410 391 ... 270 271 223]
    # [ 30  38  65 ... 225 226 169]]

    # 特徴抽出時に使ったindexの順に並んだ単語のリストを得る
    feature_names = np.array(vectorizer.get_feature_names())
    #print(feature_names)

    # indexを単語に変換
    feature_words = feature_names[index]

    # 上位の単語を出す
    list_feature_words = feature_words.tolist()
    # arrayは[w1,w2,w3,...]
    # TOP n単語出力
    n = 10
    for array, file in zip(list_feature_words,filename):
        print("file:{} words:{}".format(file,array[:n]))

    #かかった時間を出力
    elapsed_time = time.time() - start
    print("elapsed_time={}".format(elapsed_time))

if __name__ == '__main__':
    main()