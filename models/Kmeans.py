import numpy as np


class Kmeans:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.labels = []
        self.next_labels = []
        self.centroids = None

    def dist_matrix(self, data, other):
        if other is None:
            return -1
        reshape_data = data[:, np.newaxis, :]
        dif = reshape_data - other
        dists = np.sqrt(np.array(np.sum(np.abs(dif) ** 2, axis=2), dtype=np.float64))
        return dists

    def assign_labels(self, data):
        dist_cent = self.dist_matrix(data, self.centroids)
        argsrt = np.argsort(dist_cent, axis=1)
        self.labels = argsrt[:, 0]
        self.next_labels = argsrt[:, 1]

    def init_cent(self, data):
        res = np.zeros((self.num_clusters, data.shape[1]))
        res[0] = data[np.random.randint(data.shape[0])]

        for i in range(1, self.num_clusters):
            dist_cent = np.min(self.dist_matrix(data, res[:i]), axis=1)
            probs = dist_cent**2 / np.sum(dist_cent**2)
            next_cent_idx = np.random.choice(data.shape[0], p=probs)
            res[i] = data[next_cent_idx]

        return res

    def fit(self, data):
        np.random.seed(12)
        best = np.inf
        cur_cent = None
        for a in range(15):
            self.centroids = self.init_cent(data)
            for _ in range(30):
                self.assign_labels(data)
                new_centroids = self.centroids
                for i in range(self.num_clusters):
                    same_g = data[self.labels == i]
                    new_centroids[i] = np.mean(same_g, axis=0)
                self.centroids = new_centroids
            cur_sil = abs(self.silhouette_score(data) - 1)
            if best > cur_sil:
                cur_cent = self.centroids
                best = cur_sil
        self.centroids = cur_cent

    def silhouette_score(self, data):
        s = []
        n_data = data.shape[0]
        n_l = len(self.labels)
        dist_data = self.dist_matrix(data, data)
        self.assign_labels(data)
        for i in range(n_data):
            label = self.labels[i]
            grp = self.labels == label
            grp_size = np.sum(grp)
            a = np.sum(dist_data[i][grp])
            a /= grp_size - 1 if grp_size > 1 else 1

            nxt = self.next_labels[i]

            next_grp = self.labels == nxt
            next_grp_size = np.sum(next_grp)
            b = np.sum(dist_data[i][next_grp])
            b /= next_grp_size if next_grp_size > 0 else 1
            s.append((b - a) / max(a, b))
        score = np.mean(s)
        return score

    def compute_representation(self, X):
        if self.centroids is None:
            raise ValueError("The centroids are not initialized. Fit the model first.")
        dist_to_centroids = self.dist_matrix(X, self.centroids)
        return dist_to_centroids
