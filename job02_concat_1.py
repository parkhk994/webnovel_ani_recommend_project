import pandas as pd
import glob

df = pd.read_csv('crawling_data/csv_file/laftel/laftel_400_20250210_v3.csv')
df.dropna(inplace=True)
df.info()
print(df.head())

titles = []
reviews = []
book_type = []
old_title = ''
for i in range(len(df)):
    title = df.iloc[i, 0]
    if title != old_title:
        titles.append(title)
        old_title = title
        df_movie = df[(df.Titles ==  title)]
        review = ' '.join(df_movie.Reviews)
        reviews.append(review)
print(len(titles))
print(len(reviews))
df = pd.DataFrame({'titles':titles, 'reviews':reviews, 'book_type':0})
df.info()
print(df)
df.to_csv('./crawling_data/reviews_ani_laftel.csv', index=False)












