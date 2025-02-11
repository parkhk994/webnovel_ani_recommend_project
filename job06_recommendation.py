import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[:11]
    movieIdx = [i[0] for i in simScore]
    recmovieList = df_reviews.iloc[movieIdx, 0]
    return recmovieList[1:11]

df_reviews = pd.read_csv('./crawling_data/cleaned_reviews_ridi.csv')
Tfidf_matrix = mmread('./models/Tfidf_ridi_review.mtx').tocsr()
with open('models/tfidf_ridi.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 영화 index 이용
# ref_idx = 349
# print('title', df_reviews.iloc[ref_idx, 0])
# cosine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
# print(cosine_sim[0])
# print(len(cosine_sim[0]))
# recommendations = getRecommendation(cosine_sim)
# print(recommendations)

# key word 이용
embedding_model = Word2Vec.load('./models/word2vec_ridi_review.model')
keyword = '사랑'
if keyword in list(embedding_model.wv.index_to_key):
    sim_word = embedding_model.wv.most_similar(keyword, topn=10)
    words = [keyword]
    for word, _ in sim_word:
        words.append(word)
    print(words)
else :
    print('not in')
    exit()
sentence = []
count = 10
for word in words:
    sentence = sentence + [word] * count
    count -= 1
sentence = ' '.join(sentence)
print(sentence)

sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)

print(recommendation)
















