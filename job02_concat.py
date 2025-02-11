import pandas as pd
import glob

data_paths = glob.glob('./crawling_data/movie_reviews_500_movies/*')
print(data_paths)

df = pd.DataFrame()
for path in data_paths:
    df_temp = pd.read_csv(path)
    print(df_temp.head())
    titles = []
    reviews = []
    old_title = ''
    for i in range(len(df_temp)):

        title = df_temp.iloc[i, 0]
        if title != old_title:
            titles.append(title)
            old_title = title
            df_movie = df_temp[(df_temp.movie_title ==  title)]
            review = ' '.join(df_movie.review)
            reviews.append(review)
    print(len(titles))
    print(len(reviews))
    df_batch = pd.DataFrame({'titles':titles, 'reviews':reviews})
    df_batch.info()
    print(df_batch)
    df = pd.concat([df, df_batch], ignore_index=True)
df.info()
df.to_csv('./crawling_data/reviews_kinolights.csv', index=False)





















