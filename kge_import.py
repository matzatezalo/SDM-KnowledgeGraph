import torch
import pykeen
import pandas as pd
from pykeen import predict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

def import_kge():

    # Load exported TSV of all edges
    df = pd.read_csv(
        'data/kge_data/triples.tsv',
        sep='\t',
        skiprows=1,
        header=None,
        names=['head','relation','tail'],
        dtype=str,
        comment='#',       
    )

    # Drop any rows where relation is missing
    df = df.dropna(subset=["relation"])

    # Helper to get the local predicate name
    def local_name(uri: str) -> str:
        s = (uri or "").strip()
        # strip <…> if present
        if s.startswith("<") and s.endswith(">"):
            s = s[1:-1]
        if not s:
            return ""
        # split off final fragment
        return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]
    
    df['local'] = df['relation'].apply(local_name)

    # Keep only those that actually exist and that we want for the KGE
    keep_kge = {'isAuthor', 'cites', 'publishedIn'}
    df = df[df['local'].isin(keep_kge)].reset_index(drop=True)
    print("→ Number of KGE triples:", len(df))

    # Write out a triples file for PyKEEN
    out = df[['head', 'relation', 'tail']]
    out.to_csv(
        "data/kge_data/kge_triples_for_pykeen.tsv",
        sep="\t",
        index=False,
        header=False
    )
    print(f"Exported {len(out)} triples for the KGE.")

    # Create TriplesFactory
    tf = TriplesFactory.from_path(
        path='data/kge_data/kge_triples_for_pykeen.tsv'  
    )

    # Split data into train and test
    train, test = tf.split(random_state=42)

    return train, test

# Result
# result = pipeline(
#     training=train,
#     testing=test,
#     model="TransE",
#     model_kwargs=dict(
#         embedding_dim=128,
#     ),
#     training_kwargs=dict(
#         num_epochs=20
#     ),
#     optimizer_kwargs=dict(
#         lr=0.01,
#     ),
#     negative_sampler_kwargs=dict(
#         num_negs_per_pos=1,
#     ),
#     random_seed=42
# )

if __name__ == "__main__":

    train, test = import_kge()



