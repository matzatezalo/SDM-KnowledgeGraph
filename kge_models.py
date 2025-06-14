import torch
import os
import pickle

from pykeen.pipeline import pipeline
from kge_import import train, test

basic_path = "models/basic_result.pkl"

# Get most basic model
def get_basic_result(train, test):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(basic_path):
        print(f"› Loading existing pipeline from {basic_path}")
        with open(basic_path, "rb") as f:
            basic_result = pickle.load(f)
            return basic_result
    
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
        device=device
    )

    with open(basic_path, "wb") as f:
        pickle.dump(result, f)
    print("Saved trained model to", basic_path)

    return result

def most_likely_tail(paper_uri: str, relation_label: str, result):
    """
    Returns (tail_label, score, embedding) for the most
    likely completion of (head_label, relation_label, ?).
    """
    tf = result.training
    model = result.model

    # map labels to numeric IDs
    h_id = tf.entity_to_id[paper_uri]
    r_id = tf.relation_to_id[relation_label]

    # predict score
    score = model.predict_t(
        torch.tensor([[h_id, r_id]], dtype=torch.long)
    )[0]

    # Get top index + score
    top_idx = int(torch.argmax(score).item())
    top_score = float(score[top_idx].item())

    # Get tail
    inv_e = {v:k for k,v in tf.entity_to_id.items()}
    tail_label = inv_e[top_idx]

    # Get vector embedding
    emb_tensor = model.entity_representations[0](
        torch.tensor([top_idx], dtype=torch.long)
    )                    # shape = (1, embedding_dim)
    emb = emb_tensor.detach()

    return tail_label, top_score, emb

if __name__ == "__main__":
    basic_result = get_basic_result(train, test)

    all_entities = list(basic_result.training.entity_to_id.keys())

    # Choose a paper from the KG
    paper_uri = all_entities[12]

    # Predict “cites” most likely embedding vector
    cited_label, score, embedding = most_likely_tail(
        paper_uri,
        relation_label="<http://example.org/ontology/cites>",
        result=basic_result,
    )
    print("Top paper predicted to be _cited_ by", paper_uri)
    print("Predicted cited‐by:", cited_label)
    print("Score:", score)
    print("Embedding:", embedding)

