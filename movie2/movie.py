
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
print("jannat")
credits=pd.read_csv(r"credits.csv")
movies=pd.read_csv(r"movies.csv")
final=pd.merge(credits,movies)
final=final[["overview","original_title","id","genres","original_language"]]
final.isnull().sum()
final.dropna(inplace=True)

final.duplicated().value_counts()

final=final.drop_duplicates(keep="first",inplace=False)
from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(min_df=3,max_features =None, strip_accents= "unicode",analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1,3),stop_words='english')
tfv_matrix=tfv.fit_transform(final["overview"])

tfv_matrix.shape
sig=sigmoid_kernel(tfv_matrix,tfv_matrix)

index1=pd.Series(final.index,index=final["original_title"]).drop_duplicates()

def give_recommendation(title,sig=sig):
    idx=index1[title]
    sigmoid_score= list(enumerate(sig[idx]))
    sigmoid_score=sorted(sigmoid_score, key=lambda x:x[1],reverse=True)
    sigmoid_score=sigmoid_score[1:6]
    movie_indices=[i[0] for i in sigmoid_score]
    return final["original_title"].iloc[movie_indices]

print(give_recommendation("Pacific Rim"))
