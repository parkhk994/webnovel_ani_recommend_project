import pandas as pd
import glob

data_paths = glob.glob('./crawling_data/csv_file/ridi/*')
print(data_paths)

df = pd.DataFrame()
for path in data_paths:
    df_temp = pd.read_csv(path)
    print(df_temp.head())
    titles = []
    reviews = []
    book_type = []
    old_title = ''
    for i in range(len(df_temp)):

        title = df_temp.iloc[i, 0]
        if title != old_title:
            titles.append(title)
            old_title = title
            df_movie = df_temp[(df_temp.Titles ==  title)]
            # NaN 값을 빈 문자열로 대체하고 모든 값을 문자열로 변환
            df_movie['Reviews'] = df_movie['Reviews'].fillna('').astype(str)
            review = ' '.join(df_movie.Reviews)
            reviews.append(review)
    print(len(titles))
    print(len(reviews))
    df_batch = pd.DataFrame({'titles':titles, 'reviews':reviews, 'book_type':1})
    df_batch.info()
    print(df_batch)
    df = pd.concat([df, df_batch], ignore_index=True)
df.info()
df.to_csv('./crawling_data/reviews_novel_ridi_2.csv', index=False)
