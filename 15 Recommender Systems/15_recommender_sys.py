import numpy as np

np.random.seed(14)

d = np.random.randint(low=1, high=5, size=(5, 10))
print(d)

# Pairwise correlation coef
print(np.corrcoef(d[0], d[1]))

# Row-wise (user to user) and column-wise (movie to movie) correlation coefs
person_sim = np.corrcoef(d, rowvar=True)
movie_sim = np.corrcoef(d, rowvar=False)

print(person_sim)
print(movie_sim)

# Most similar to movie 0
np.argsort(movie_sim[0])[::-1]
np.argsort(movie_sim[0])[:4:-1] # top 5 most similar movies
np.sort(movie_sim[0])[:4:-1]    # sim scores for top 5 most similar movies
