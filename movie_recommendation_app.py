import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl, QStringListModel
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle

form_window = uic.loadUiType('./movie_recommendation.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_ridi_review.mtx').tocsr()
        with open('models/tfidf_ridi.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_ridi_review.model')

        self.df_reviews = pd.read_csv('./crawling_data/cleaned_reviews_ridi.csv')
        self.titles = list(self.df_reviews.titles)
        self.titles.sort()
        for title in self.titles:
            self.cb_title.addItem(title)

        model = QStringListModel()
        model.setStringList((self.titles))
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

        self.cb_title.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommend.clicked.connect(self.btn_slot)

        # Make labels clickable
        self.lbl_recommendation_0.setOpenExternalLinks(True)
        self.lbl_recommendation_1.setOpenExternalLinks(True)

        #초기 이미지 파일 경로와  파일 필터를 설정
        self.path1 = 'C:/work/webnovel_ani_recommend_project/imgs/01_laftel.png'
        self.path2 = 'C:/work/webnovel_ani_recommend_project/imgs/02_ridi.jpg'
        self.path3 = 'C:/work/webnovel_ani_recommend_project/imgs/03_left.jpg'

        # 라벨 크기에 맞게 이미지를 크기 조정
        pixmap1 = QPixmap(self.path1).scaled(self.lbl_img_01.size(), aspectRatioMode= Qt.KeepAspectRatio)
        pixmap2 = QPixmap(self.path2).scaled(self.lbl_img_02.size(), aspectRatioMode= Qt.KeepAspectRatio)
        pixmap3 = QPixmap(self.path3).scaled(self.lbl_img_03.size(), aspectRatioMode=Qt.KeepAspectRatio)

        #이미지를 QPixmap 객체로 변환하여 lbl_image라벨에 표시
        self.lbl_img_01.setPixmap(pixmap1)
        self.lbl_img_02.setPixmap(pixmap2)
        self.lbl_img_03.setPixmap(pixmap3)

        
        # No need for separate open_recommendation_page method

    def btn_slot(self):
        keyword = self.le_keyword.text()
        if keyword in self.titles:
            recommendation = self.recommendation_by_title(keyword)
        else:
            recommendation = self.recommendation_by_keyword(keyword)
        
        if recommendation:
            # Separate recommendations by book_type and create search URLs
            type_0_recs = [
                f'<a href="https://laftel.net/search?keyword={rec}" style="color: white; text-decoration: none;">{self.truncate_title(rec)}</a><br><br>'
                for rec in recommendation 
                if self.df_reviews[self.df_reviews.titles == rec].book_type.values[0] == 0
            ]
            type_1_recs = [
                f'<a href="https://ridibooks.com/search?q={rec}&adult_exclude=y" style="color: white; text-decoration: none;">{self.truncate_title(rec)}</a><br><br>'
                for rec in recommendation 
                if self.df_reviews[self.df_reviews.titles == rec].book_type.values[0] == 1
            ]
            
            # Display recommendations
            self.lbl_recommendation_0.setText(''.join(type_0_recs))
            self.lbl_recommendation_1.setText(''.join(type_1_recs))

    def combobox_slot(self):
        title = self.cb_title.currentText()
        print(title)
        recommendation = self.recommendation_by_title(title)
        
        if recommendation:
            # Separate recommendations by book_type and create search URLs
            type_0_recs = [
                f'<a href="https://laftel.net/search?keyword={rec}" style="color: white; text-decoration: none;">{self.truncate_title(rec)}</a><br>'
                for rec in recommendation 
                if self.df_reviews[self.df_reviews.titles == rec].book_type.values[0] == 0
            ]
            type_1_recs = [
                f'<a href="https://ridibooks.com/search?q={rec}&adult_exclude=y" style="color: white; text-decoration: none;">{self.truncate_title(rec)}</a><br>'
                for rec in recommendation 
                if self.df_reviews[self.df_reviews.titles == rec].book_type.values[0] == 1
            ]
            
            # Display recommendations
            self.lbl_recommendation_0.setText(''.join(type_0_recs))
            self.lbl_recommendation_1.setText(''.join(type_1_recs))

    def truncate_title(self, title, max_length=20):
        return (title[:max_length] + '...') if len(title) > max_length else title

    def recommendation_by_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews.titles == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        return recommendation

    def recommendation_by_keyword(self, keyword):
        try:
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
        except:
            self.lbl_recommendation_0.setText('제가 모르는 단어에요 ㅠㅠ')
            return 0
        words = [keyword]
        for word, _ in sim_word:
            words.append(word)
        print(words)
        sentence = []
        count = 10
        for word in words:
            sentence = sentence + [word] * count
            count -= 1
        sentence = ' '.join(sentence)
        print(sentence)
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        return recommendation


    def getRecommendation(self, cosine_sim):
        # Get the book_type of the input title
        input_book_type = self.df_reviews.book_type.iloc[0] if len(cosine_sim) > 0 else None
        
        # Prepare similarity scores
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]  # Exclude the input title itself
        
        # Separate recommendations by book_type
        same_type_recs = []
        diff_type_recs = []
        
        for score in sim_scores:
            idx = score[0]
            book_type = self.df_reviews.book_type.iloc[idx]
            rec_title = self.df_reviews.titles.iloc[idx]
            
            if book_type == input_book_type and len(same_type_recs) < 5:
                same_type_recs.append(rec_title)
            elif book_type != input_book_type and len(diff_type_recs) < 5:
                diff_type_recs.append(rec_title)
            
            # Stop if we have 5 of each type
            if len(same_type_recs) == 5 and len(diff_type_recs) == 5:
                break
        
        # Combine recommendations
        recommendations = same_type_recs + diff_type_recs
        return recommendations


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())
