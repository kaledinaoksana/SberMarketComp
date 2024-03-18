
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics


def load_csv(filename, path):
    """
    Load CSV file.
    """
    file = path+filename+'.csv'
    return pd.read_csv(file, low_memory=False)


def some_ids_preparation_to_als(user_item_matrix):

    userids = user_item_matrix.index.values
    itemids = user_item_matrix.columns.values
    
    matrix_userids = np.arange(len(userids))
    matrix_itemids = np.arange(len(itemids))
    
    id_to_itemid = dict(zip(matrix_itemids, itemids))
    id_to_userid = dict(zip(matrix_userids, userids))
    
    itemid_to_id = dict(zip(itemids, matrix_itemids))
    userid_to_id = dict(zip(userids, matrix_userids))
    
    return itemid_to_id, userid_to_id, id_to_itemid, id_to_userid


def figure_umap_embeddings(model_als, umap_emb, name):

    plt.figure(figsize=(10, 7))
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], s=1)  # Рассеиваем точки
    plt.title(name)  # Заголовок
    plt.xlabel('UMAP Component 1')  # Метка оси X
    plt.ylabel('UMAP Component 2')  # Метка оси Y
    plt.grid(True)  # Включаем сетку

    model_info = f"""
    Model: ALS, 
    Factors: {model_als.factors} 
    Regularization: {model_als.regularization}
    Iterations: {model_als.iterations}
    """
    
    # Добавление информации о модели в квадратике на графике
    if model_info:
        plt.text(0.77, 0.05, model_info, ha='left', va='bottom', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=1))
    
    # Определение save_path, чтобы избежать ошибки UnboundLocalError
    save_path = None

    if name == 'UMAP Visualization of User Embeddings':
        name = f"ALS_f{model_als.factors}_r{model_als.regularization}_i{model_als.iterations}"
        save_path = f'figures/{name}.png'

    # Сохранение графика в файл, если указан путь для сохранения
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show() 


# PRINT CLUSTER
def print_clusters(X, labels, core_samples_mask,n_clusters_):
    size = 10
    s = 7
    plt.figure(figsize=(20, 10))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        # noize
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        if len(xy) > 0:
            centroid_x = xy[:, 0].mean()
            centroid_y = xy[:, 1].mean()
        else:
            centroid_x = 0
            centroid_y = 0
            
        xy = X[class_member_mask & core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=s, c=[col], label='Cluster %d' % k)

    
        # Добавляем метку кластера
        if k != -1:
            a=1
            centroid_x = xy[:, 0].mean()
            centroid_y = xy[:, 1].mean()
            xx=centroid_x
            yy=centroid_y+a
            plt.plot([centroid_x, xx], [centroid_y, yy], color='gray', linestyle='--')
            plt.text(xx, yy, str(k), fontsize=size, color='black',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))
        else:
            xy = X[class_member_mask & ~core_samples_mask]
            plt.scatter(xy[:, 0], xy[:, 1], s=s, c=[col])

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    pass

def compute_cosine_similarity(cluster_embeddings):
        similarity_matrix = cosine_similarity(cluster_embeddings)
        average_similarity = similarity_matrix.mean()
        return average_similarity

#DBSCAN
def print_dbscan(X, n_clusters):

    db = DBSCAN(eps=0.3, min_samples=n_clusters).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    cos_sim = {}

    for cluster_id in range(n_clusters_):
        cluster_points = X[labels == cluster_id]
        cos_sim[cluster_id] = compute_cosine_similarity(cluster_points)

    print(core_samples_mask)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    print_clusters(X, labels, core_samples_mask, n_clusters_)

    return labels, n_clusters_