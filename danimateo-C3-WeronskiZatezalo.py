import itertools
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# def import_kge(path: str="data/kge_data/kge_triples_for_pykeen.tsv") -> TriplesFactory:
#     return TriplesFactory.from_path(path, create_inverse_triples=False)

def run_experiment(tf, model_name, emb_dim, num_negs, num_epochs=20, lr=0.05, seed=42):
    result = pipeline(
        training=tf,
        testing=tf,
        model=model_name,
        model_kwargs=dict(embedding_dim=emb_dim),
        training_kwargs=dict(num_epochs=num_epochs),
        optimizer_kwargs=dict(lr=lr),
        negative_sampler_kwargs=dict(num_negs_per_pos=num_negs),
        random_seed=seed,
    )

    # Get evaluation metrics
    metrics = result.metric_results.to_flat_dict()

    # Return training parameters and hyperparameters
    return {
        "model": model_name,
        "emb_dim": emb_dim,
        "neg_per_pos": num_negs,
        "mrr": metrics["both.realistic.inverse_harmonic_mean_rank"],
        "mdr": metrics["both.realistic.median_rank"],
        "h@1": metrics["both.realistic.hits_at_1"],
        "h@10": metrics["both.realistic.hits_at_10"],
    }

if __name__ == "__main__":
    tf = TriplesFactory.from_path("kgesmall.tsv")

    experiments = []
    # Model names, embedding dimensions and number of negative samples per positive
    for model_name, emb_dim, neg in itertools.product(
        ["TransE", "DistMult", "ComplEx"],   
        [64, 128],                            
        [1, 5],                               
    ):
        print(f"\n→ training {model_name} | dim={emb_dim} | negs={neg}")
        stats = run_experiment(tf, model_name, emb_dim, neg, num_epochs=20)
        print(f"   → MRR={stats['mrr']:.4f},  H@10={stats['h@10']:.4f}")
        experiments.append(stats)

    df = pd.DataFrame(experiments).sort_values("mrr", ascending=False)
    df.to_csv("kge_experiment_results.csv", index=False)