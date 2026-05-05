from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(t1, t2):
    vec = TfidfVectorizer()
    v = vec.fit_transform([t1, t2])
    return cosine_similarity(v[0], v[1])[0][0]