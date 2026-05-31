import zipfile
import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt

ZIP_PATH = 'ml-latest-small.zip'
RANDOM_SEED = 42
TEST_SIZE = 0.2
KFOLDS = 4
ALS_ITERS = 5
ALS_FACTORS = [5, 10, 15]
ALS_REGS = [0.001, 0.01, 0.1, 1, 10]
VARIANT_GENRES = ['Drama', 'Comedy', 'Musical']
TARGET_USER_ID = None


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_test_split_df(df, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def kfold_indices(n, k=4, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def build_index_maps(train_df):
    users = np.sort(train_df['userId'].unique())
    items = np.sort(train_df['movieId'].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {m: i for i, m in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item


def prepare_user_history(train_df, user2idx, item2idx):
    n_users = len(user2idx)
    user_items = [[] for _ in range(n_users)]
    user_ratings = [[] for _ in range(n_users)]
    for r in train_df.itertuples(index=False):
        u = user2idx[r.userId]
        i = item2idx[r.movieId]
        user_items[u].append(i)
        user_ratings[u].append(float(r.rating))
    user_items = [np.array(x, dtype=np.int32) for x in user_items]
    user_ratings = [np.array(x, dtype=np.float64) for x in user_ratings]
    return user_items, user_ratings


def prepare_item_history(train_df, user2idx, item2idx):
    n_items = len(item2idx)
    item_users = [[] for _ in range(n_items)]
    item_ratings = [[] for _ in range(n_items)]
    for r in train_df.itertuples(index=False):
        u = user2idx[r.userId]
        i = item2idx[r.movieId]
        item_users[i].append(u)
        item_ratings[i].append(float(r.rating))
    item_users = [np.array(x, dtype=np.int32) for x in item_users]
    item_ratings = [np.array(x, dtype=np.float64) for x in item_ratings]
    return item_users, item_ratings


def item_based_cf_predict_batch(test_df, train_df, user2idx, item2idx):
    n_users = len(user2idx)
    n_items = len(item2idx)
    mu = float(train_df['rating'].mean())

    item_user = np.zeros((n_items, n_users), dtype=np.float32)
    for r in train_df.itertuples(index=False):
        item_user[item2idx[r.movieId], user2idx[r.userId]] = float(r.rating)
    norms = np.linalg.norm(item_user, axis=1)
    norms[norms == 0] = 1.0
    item_user_norm = item_user / norms[:, None]

    user_items, user_ratings = prepare_user_history(train_df, user2idx, item2idx)

    test_by_user = defaultdict(list)
    for r in test_df.itertuples(index=False):
        u = user2idx.get(r.userId)
        i = item2idx.get(r.movieId)
        if u is None or i is None:
            continue
        test_by_user[u].append((i, float(r.rating)))

    preds = []
    truth = []
    for u, lst in test_by_user.items():
        hist = user_items[u]
        hist_r = user_ratings[u]
        y_true = np.array([r for _, r in lst], dtype=np.float64)
        if hist.size == 0:
            preds.extend([mu] * len(lst))
            truth.extend(y_true.tolist())
            continue
        H = item_user_norm[hist]  # h x d
        test_items = np.array([i for i, _ in lst], dtype=np.int32)
        V = item_user_norm[test_items]  # t x d
        S = V @ H.T  # t x h
        S[S < 0] = 0.0
        denom = S.sum(axis=1)
        num = S @ hist_r
        p = np.where(denom > 0, num / denom, mu)
        preds.extend(p.tolist())
        truth.extend(y_true.tolist())

    return float(np.sqrt(np.mean((np.array(truth) - np.array(preds)) ** 2)))


def fit_als(train_df, user2idx, item2idx, k=10, reg=0.1, n_iters=5, seed=0):
    n_users = len(user2idx)
    n_items = len(item2idx)
    user_items, user_ratings = prepare_user_history(train_df, user2idx, item2idx)
    item_users, item_ratings = prepare_item_history(train_df, user2idx, item2idx)

    rng = np.random.default_rng(seed)
    X = 0.01 * rng.standard_normal((n_users, k))
    Y = 0.01 * rng.standard_normal((n_items, k))
    I = np.eye(k)

    for _ in range(n_iters):
        for u in range(n_users):
            items_u = user_items[u]
            if items_u.size == 0:
                continue
            Y_u = Y[items_u]
            A = Y_u.T @ Y_u + reg * I
            b = Y_u.T @ user_ratings[u]
            X[u] = np.linalg.solve(A, b)
        for i in range(n_items):
            users_i = item_users[i]
            if users_i.size == 0:
                continue
            X_i = X[users_i]
            A = X_i.T @ X_i + reg * I
            b = X_i.T @ item_ratings[i]
            Y[i] = np.linalg.solve(A, b)
    return X, Y


def als_rmse(train_df, val_df, k=10, reg=0.1, n_iters=5, seed=0):
    user2idx, item2idx, _, _ = build_index_maps(train_df)
    X, Y = fit_als(train_df, user2idx, item2idx, k=k, reg=reg, n_iters=n_iters, seed=seed)
    y_true = []
    y_pred = []
    covered = 0
    for r in val_df.itertuples(index=False):
        u = user2idx.get(r.userId)
        i = item2idx.get(r.movieId)
        if u is None or i is None:
            continue
        covered += 1
        y_true.append(float(r.rating))
        y_pred.append(float(X[u] @ Y[i]))
    if covered == 0:
        return np.nan, 0.0
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))), covered / len(val_df)


def recommend_for_user_cf(user_id, train_df, movies_df, user2idx, item2idx, top_n=10):
    n_users = len(user2idx)
    n_items = len(item2idx)
    mu = float(train_df['rating'].mean())

    item_user = np.zeros((n_items, n_users), dtype=np.float32)
    for r in train_df.itertuples(index=False):
        item_user[item2idx[r.movieId], user2idx[r.userId]] = float(r.rating)
    norms = np.linalg.norm(item_user, axis=1)
    norms[norms == 0] = 1.0
    item_user_norm = item_user / norms[:, None]

    u = user2idx[user_id]
    hist = []
    hist_r = []
    for r in train_df.itertuples(index=False):
        if r.userId == user_id:
            hist.append(item2idx[r.movieId])
            hist_r.append(float(r.rating))
    hist = np.array(hist, dtype=np.int32)
    hist_r = np.array(hist_r, dtype=np.float32)

    rated = set(train_df.loc[train_df['userId'] == user_id, 'movieId'].tolist())
    candidates = [m for m in item2idx.keys() if m not in rated]
    cand_idx = np.array([item2idx[m] for m in candidates], dtype=np.int32)

    H = item_user_norm[hist]
    V = item_user_norm[cand_idx]
    S = V @ H.T
    S[S < 0] = 0.0
    denom = S.sum(axis=1)
    num = S @ hist_r
    scores = np.where(denom > 0, num / denom, mu)

    out = pd.DataFrame({'movieId': candidates, 'score': scores})
    out = out.merge(movies_df[['movieId', 'title']], on='movieId', how='left').sort_values('score', ascending=False)
    return out.head(top_n).reset_index(drop=True)


def recommend_for_user_als(user_id, train_df, movies_df, user2idx, item2idx, X, Y, top_n=10):
    rated = set(train_df.loc[train_df['userId'] == user_id, 'movieId'].tolist())
    candidates = [m for m in item2idx.keys() if m not in rated]
    cand_idx = np.array([item2idx[m] for m in candidates], dtype=np.int32)
    u = user2idx[user_id]
    scores = X[u] @ Y[cand_idx].T
    out = pd.DataFrame({'movieId': candidates, 'score': scores})
    out = out.merge(movies_df[['movieId', 'title']], on='movieId', how='left').sort_values('score', ascending=False)
    return out.head(top_n).reset_index(drop=True)


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        movies = pd.read_csv(z.open('ml-latest-small/movies.csv'))
        ratings = pd.read_csv(z.open('ml-latest-small/ratings.csv'))

    # Task 1
    movies_expl = movies.assign(genre=movies['genres'].str.split('|')).explode('genre')
    movie_stats = ratings.groupby('movieId').agg(
        rating_count=('rating', 'size'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    genre_movies = movies_expl[movies_expl['genre'].isin(VARIANT_GENRES)].drop_duplicates(['movieId', 'genre']).merge(movie_stats, on='movieId', how='left')
    genre_counts = genre_movies.groupby('genre')['movieId'].nunique().reindex(VARIANT_GENRES)

    task1 = {}
    for g in VARIANT_GENRES:
        d = genre_movies[genre_movies['genre'] == g].copy()
        top_count = d.sort_values(['rating_count', 'title'], ascending=[False, True]).head(10)[['movieId', 'title', 'rating_count', 'avg_rating']]
        low_count = d[d['rating_count'] > 10].sort_values(['rating_count', 'title'], ascending=[True, True]).head(10)[['movieId', 'title', 'rating_count', 'avg_rating']]
        top_avg = d[d['rating_count'] > 10].sort_values(['avg_rating', 'rating_count', 'title'], ascending=[False, True, True]).head(10)[['movieId', 'title', 'rating_count', 'avg_rating']]
        low_avg = d[d['rating_count'] > 10].sort_values(['avg_rating', 'rating_count', 'title'], ascending=[True, True, True]).head(10)[['movieId', 'title', 'rating_count', 'avg_rating']]
        task1[g] = {
            'top_count': top_count,
            'low_count': low_count,
            'top_avg': top_avg,
            'low_avg': low_avg,
        }

    # Task 2
    train_init, test = train_test_split_df(ratings, test_size=TEST_SIZE, seed=RANDOM_SEED)
    global_mean = float(train_init['rating'].mean())
    baseline_rmse = rmse(test['rating'], np.full(len(test), global_mean))
    user2idx, item2idx, _, _ = build_index_maps(train_init)
    cf_rmse = item_based_cf_predict_batch(test, train_init, user2idx, item2idx)

    folds = kfold_indices(len(train_init), k=KFOLDS, seed=RANDOM_SEED)
    cv_rows = []
    best = None
    best_score = np.inf
    for k in ALS_FACTORS:
        for reg in ALS_REGS:
            fold_scores = []
            coverages = []
            for fold_id in range(KFOLDS):
                val_idx = folds[fold_id]
                train_idx = np.concatenate([folds[j] for j in range(KFOLDS) if j != fold_id])
                tr = train_init.iloc[train_idx].reset_index(drop=True)
                va = train_init.iloc[val_idx].reset_index(drop=True)
                score, cov = als_rmse(tr, va, k=k, reg=reg, n_iters=ALS_ITERS, seed=RANDOM_SEED + fold_id)
                fold_scores.append(score)
                coverages.append(cov)
            mean_score = float(np.nanmean(fold_scores))
            mean_cov = float(np.mean(coverages))
            cv_rows.append({'factors': k, 'reg': reg, 'rmse_cv': mean_score, 'coverage': mean_cov})
            if mean_score < best_score:
                best_score = mean_score
                best = {'factors': k, 'reg': reg, 'rmse_cv': mean_score, 'coverage': mean_cov}

    cv_df = pd.DataFrame(cv_rows).sort_values(['rmse_cv', 'factors', 'reg']).reset_index(drop=True)

    X_best, Y_best = fit_als(train_init, user2idx, item2idx, k=best['factors'], reg=best['reg'], n_iters=ALS_ITERS, seed=RANDOM_SEED)
    als_test_pred = []
    als_test_true = []
    als_cov = 0
    for r in test.itertuples(index=False):
        u = user2idx.get(r.userId)
        i = item2idx.get(r.movieId)
        if u is None or i is None:
            als_test_pred.append(global_mean)
        else:
            als_cov += 1
            als_test_pred.append(float(X_best[u] @ Y_best[i]))
        als_test_true.append(float(r.rating))
    als_test_rmse = rmse(als_test_true, als_test_pred)

    active_user = train_init['userId'].value_counts().idxmax()
    cf_rec = recommend_for_user_cf(active_user, train_init, movies, user2idx, item2idx, top_n=10)
    als_rec = recommend_for_user_als(active_user, train_init, movies, user2idx, item2idx, X_best, Y_best, top_n=10)
    overlap = len(set(cf_rec['movieId']) & set(als_rec['movieId']))

    lines = []
    lines.append('Recommender Systems and Spark MLlib')
    lines.append('')
    lines.append('Variants: Task 1 — Drama / Comedy / Musical; Task 2 — item similarity; Task 3 — ALS.')
    lines.append('')
    lines.append('Task 1. Genre analysis')
    lines.append('Genre counts:')
    lines.append(genre_counts.to_string())
    lines.append('')
    for g in VARIANT_GENRES:
        lines.append(f'Genre: {g}')
        lines.append('Top 10 by rating count:')
        lines.append(task1[g]['top_count'].to_string(index=False))
        lines.append('Bottom 10 by rating count (rating_count > 10):')
        lines.append(task1[g]['low_count'].to_string(index=False))
        lines.append('Top 10 by average rating (rating_count > 10):')
        lines.append(task1[g]['top_avg'].to_string(index=False))
        lines.append('Bottom 10 by average rating (rating_count > 10):')
        lines.append(task1[g]['low_avg'].to_string(index=False))
        lines.append('')

    lines.append('Task 2. Collaborative filtering')
    lines.append(f'Train mean rating = {global_mean:.6f}')
    lines.append(f'Baseline RMSE (predict train mean for all test rows) = {baseline_rmse:.6f}')
    lines.append(f'Item-based CF RMSE on test = {cf_rmse:.6f}')
    lines.append('')

    lines.append('Task 3. ALS cross-validation')
    lines.append('CV results sorted by RMSE:')
    lines.append(cv_df.to_string(index=False))
    lines.append('')
    lines.append('Best ALS configuration:')
    lines.append(str(best))
    lines.append(f'ALS RMSE on test = {als_test_rmse:.6f}')
    lines.append(f'ALS test coverage with known user/item pairs = {als_cov/len(test):.4%}')
    lines.append('')
    lines.append(f'Recommendation comparison for user {active_user} (most active user in train_init):')
    lines.append('CF top-10:')
    lines.append(cf_rec.to_string(index=False))
    lines.append('ALS top-10:')
    lines.append(als_rec.to_string(index=False))
    lines.append(f'Overlap in top-10 = {overlap} movies')

    report = '\n'.join(lines)

if __name__ == '__main__':
    main()
