import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from collections import defaultdict
from sklearn.metrics import silhouette_score
from danimateo_C1_WeronskiZatezalo import import_kge

from sklearn.manifold import TSNE
import seaborn as sns

def import_kge(path="data/kge_data/kge_triples_for_pykeen.tsv"):
    tf = TriplesFactory.from_path(path)
    train, _ = tf.split(random_state=42)
    return train

def get_model(train, model_path):
    # if we already have a pipeline result, load it
    if os.path.exists(model_path):
        print(f"→ Loading saved model from {model_path}")
        with open(model_path, "rb") as f:
            result = pickle.load(f)
    else:
        print("→ Training DistMult model …")
        result = pipeline(
            training=train,
            model="DistMult",
            model_kwargs=dict(embedding_dim=128),
            training_kwargs=dict(num_epochs=50),
            optimizer_kwargs=dict(lr=0.05),
            negative_sampler_kwargs=dict(num_negs_per_pos=1),
            random_seed=42,
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(result, f)
        print(f"→ Model saved to {model_path}")
    return result

def cluster_authors(result, n_clusters=3):
    """
    Extract all author embeddings (entities whose URI contains '/person/'),
    cluster them with KMeans, and return a dict URI → cluster_label.
    """
    # Get all entity URIs that look like authors
    ent2id = result.training.entity_to_id
    author_uris = [uri for uri in ent2id if "/person/" in uri.lower()]
    author_ids  = [ent2id[uri]       for uri in author_uris]

    # Batch-fetch their embedding vectors
    emb_tensor = result.model.entity_representations[0](
        torch.tensor(author_ids, dtype=torch.long)
    ).detach().cpu().numpy()

    # Cluster with KMeans algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(emb_tensor)
    score = silhouette_score(emb_tensor, labels)

    # Build mapping URI → cluster
    return dict(zip(author_uris, labels)), score, emb_tensor, labels, kmeans

# Get top 3 authors in clusters
def describe_clusters(clusters, top_k=3):
    by_label = defaultdict(list)
    for uri, lab in clusters.items():
        by_label[lab].append(uri)
    for lab, uris in sorted(by_label.items()):
        print(f"Cluster {lab} (size={len(uris)}):")
        for u in uris[:top_k]:
            print("  ", u)
        if len(uris) > top_k:
            print("   …")
        print()


# Project the author embeddings into 2D and plot clusters
def plot_clusters(emb_tensor, labels):
    p2 = PCA(2).fit_transform(emb_tensor)
    plt.figure(figsize=(6,6))
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(p2[mask,0], p2[mask,1], label=f"C{lab}", s=10)
    plt.legend(markerscale=2)
    plt.title("Author embeddings (PCA 2D) by cluster")
    plt.show()


if __name__ == "__main__":
    
    train = import_kge()
    model_path = "models/basic_result.pkl"

    # Train or load TransE
    result = get_model(train, model_path)

    # Get clusters and cluster information 
    clusters, score, emb_tensor, labels, kmeans = cluster_authors(result, n_clusters=3)

    # Get top 3 authors of clusters to display them
    describe_clusters(clusters, top_k=3)
    
    # Print silhouette score
    print(f"Silhouette score = {score:.3f}")

    # Plot clusters
    plot_clusters(emb_tensor, labels)


    

