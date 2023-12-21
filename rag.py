from getpass import getpass
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pandas as pd

OPENAI_API_KEY = ''

embeddings = OpenAIEmbeddings()

# CSVファイルの読み込み
data = pd.read_csv('text.csv', header=None, names=['text', 'author'])

etable = embeddings.embed_documents(data['text'])

# 質問の埋め込み
question_text = "親譲りの無鉄砲で小供の時から損ばかりしている。"
question_vector = embeddings.embed_query(question_text)

# 類似度の計算（例：コサイン類似度を使用）
from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity([question_vector], etable)

# 最も類似度が高い小説家の特定
most_similar_index = similarity_scores.argmax()
most_similar_author = data.iloc[most_similar_index]['author']

# 結果の出力
print(f"この質問に最も近いスタイルの小説家は {most_similar_author} です。")