from pykeen.pipeline import pipeline

from kge_import import train, test

# Get most basic model
def get_basic_result(train, test):
    """Train a quick TransE model and return the pipeline result."""
    result = pipeline(
        training=train,
        testing=test,
        model='TransE',
        model_kwargs=dict(embedding_dim=128),
        training_kwargs=dict(num_epochs=50),
        optimizer_kwargs=dict(lr=0.01),
        negative_sampler_kwargs=dict(num_negs_per_pos=1),
        random_seed=42,
    )

    return result

def most_likely_tails(paper_uri: str, relation_label: str, result, k=3):
    """
    Returns the top‐k predicted tail entities for (paper, relation_label, ?).
    """
    return get_tail_prediction_df(
        model=result.model,
        mapped_triples=result.training,
        head_label=paper_uri,
        relation_label=relation_label,
        k=k,
    )

if __name__ == "__main__":
    basic_result = get_basic_result(train, test)

    # Choose a paper from the KG
    paper_uri = "http://example.org/ontology/paper/1f3a2c78393eb079242e4b25f6e1f28352c5ce8f"

    # 3) predict “cites” tails
    cited = most_likely_tails(
        paper_uri,
        relation_label="ex:cites",
        pipeline_result=basic_result,
        k=3,
    )
    print("Top 3 papers predicted to be _cited_ by", paper_uri)
    print(cited)

