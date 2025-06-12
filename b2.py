import json
import os
import glob
from urllib.parse import quote_plus

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD

# --- Setup: Namespaces ---
# Use the same namespace as our TBox for consistency.
EX = Namespace("http://example.org/ontology/")

# A set to keep track of URIs we've already processed to avoid duplicate name/type triples.
# This is a simple in-memory way to handle deduplication for this script run.
processed_uris = set()


def safe_url_string(s):
    """
    Creates a URL-safe string for use in URIs.
    Replaces spaces with underscores and handles other special characters.
    """
    if not s:
        return ""
    # Using quote_plus to handle a wide range of characters safely.
    return quote_plus(s.replace(" ", "_"))


def process_paper_entry(entry, g):
    """
    Processes a single paper entry from the JSON data and adds triples to the graph.

    Args:
        entry (dict): A dictionary representing one paper.
        g (Graph): The rdflib Graph to add triples to.
    """
    # --- 1. Core Paper Information ---
    paper_id = entry.get("paperId")
    if not paper_id:
        return  # Skip if there's no paperId, as it's our primary identifier.

    paper_uri = EX[f"paper/{paper_id}"]

    # Add paper type and basic info only once
    if paper_uri not in processed_uris:
        g.add((paper_uri, RDF.type, EX.Paper))
        if entry.get("title"):
            g.add((paper_uri, EX.hasName, Literal(entry["title"])))
        if entry.get("abstract"):
            g.add((paper_uri, EX.hasAbstract, Literal(entry["abstract"], lang="en")))
        processed_uris.add(paper_uri)

    # --- 2. Authors ---
    authors = entry.get("authors", [])
    for author in authors:
        author_id = author.get("authorId")
        if author_id:
            author_uri = EX[f"person/{author_id}"]
            # Define author type and name only once
            if author_uri not in processed_uris:
                g.add((author_uri, RDF.type, EX.Person))
                if author.get("name"):
                    g.add((author_uri, EX.hasName, Literal(author["name"])))
                processed_uris.add(author_uri)
            # Create the link between the person and the paper
            g.add((author_uri, EX.isAuthor, paper_uri))

    # --- 3. Topics (Fields of Study) ---
    fields = entry.get("fieldsOfStudy", [])
    if fields:
        for field in fields:
            if field:
                topic_uri = EX[f"topic/{safe_url_string(field)}"]
                # Define topic type and name only once
                if topic_uri not in processed_uris:
                    g.add((topic_uri, RDF.type, EX.Topic))
                    g.add((topic_uri, EX.hasField, Literal(field)))
                    processed_uris.add(topic_uri)
                # Link paper to topic
                g.add((paper_uri, EX.isAbout, topic_uri))

    # --- 4. Citations ---
    citations = entry.get("citations", [])
    for citation in citations:
        cited_paper_id = citation.get("paperId")
        if cited_paper_id:
            cited_paper_uri = EX[f"paper/{cited_paper_id}"]
            # We don't know the type of the cited paper for sure from this data,
            # but per RDFS range rules, using ex:cites implies it's a Paper.
            # We can also add its type and title if we haven't seen it before.
            if cited_paper_uri not in processed_uris:
                g.add((cited_paper_uri, RDF.type, EX.Paper))
                if citation.get("title"):
                    g.add((cited_paper_uri, EX.hasName, Literal(citation.get("title"))))
                # Note: We are adding it to processed_uris to avoid re-adding the name,
                # but a full run over all files would eventually process this paper's full entry.
                processed_uris.add(cited_paper_uri)
            g.add((paper_uri, EX.cites, cited_paper_uri))

    # --- 5. Publication Venue (Journal/Conference) ---
    pub_venue = entry.get("publicationVenue")
    if pub_venue and pub_venue.get("id"):
        venue_id = pub_venue["id"]
        venue_type = pub_venue.get("type")  # "journal" or "conference"

        # The publication is a specific instance (Volume or Edition)
        publication_instance_uri = None

        if venue_type == "journal" and entry.get("journal", {}).get("volume"):
            journal_uri = EX[f"journal/{venue_id}"]
            volume_num = entry["journal"]["volume"]
            publication_instance_uri = EX[
                f"volume/{venue_id}-v{safe_url_string(volume_num)}"
            ]

            if journal_uri not in processed_uris:
                g.add((journal_uri, RDF.type, EX.Journal))
                g.add((journal_uri, EX.hasName, Literal(pub_venue.get("name"))))
                processed_uris.add(journal_uri)

            if publication_instance_uri not in processed_uris:
                g.add((publication_instance_uri, RDF.type, EX.Volume))
                g.add(
                    (publication_instance_uri, EX.volumeBelongsToJournal, journal_uri)
                )
                g.add((publication_instance_uri, EX.hasId, Literal(volume_num)))
                processed_uris.add(publication_instance_uri)

        elif venue_type == "conference" and entry.get("year"):
            # This logic assumes a conference has one main edition per year
            event_uri = EX[f"conference/{venue_id}"]
            year_val = entry["year"]
            publication_instance_uri = EX[f"edition/{venue_id}-{year_val}"]

            if event_uri not in processed_uris:
                g.add((event_uri, RDF.type, EX.Conference))
                g.add((event_uri, EX.hasName, Literal(pub_venue.get("name"))))
                processed_uris.add(event_uri)

            if publication_instance_uri not in processed_uris:
                g.add((publication_instance_uri, RDF.type, EX.Edition))
                g.add((publication_instance_uri, EX.editionBelongsToEvent, event_uri))
                # Note: City info is not in the JSON, so ex:hostedIn cannot be added.
                processed_uris.add(publication_instance_uri)

        # Link paper to the publication instance (Volume or Edition)
        if publication_instance_uri:
            g.add((paper_uri, EX.publishedIn, publication_instance_uri))

            # Link publication instance to year
            if entry.get("year"):
                year_val = entry["year"]
                year_uri = EX[f"year/{year_val}"]
                if year_uri not in processed_uris:
                    g.add((year_uri, RDF.type, EX.Year))
                    g.add(
                        (
                            year_uri,
                            EX.yearValue,
                            Literal(str(year_val), datatype=XSD.gYear),
                        )
                    )
                    processed_uris.add(year_uri)
                g.add((publication_instance_uri, EX.publishedInYear, year_uri))


def create_abox_from_json(json_folder_path, output_file):
    """
    Reads all JSON files in a folder, processes them, and saves the resulting
    RDF graph to a Turtle file.

    Args:
        json_folder_path (str): The path to the folder containing JSON files.
        output_file (str): The name of the file to save the RDF graph to.
    """
    print(f"Starting ABox generation from JSON files in: {json_folder_path}")
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("ex", EX)

    # Find all json files in the specified directory
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))

    if not json_files:
        print(f"Warning: No JSON files found in '{json_folder_path}'.")
        # Create some dummy files for demonstration if none are found.
        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path)
        # with open(os.path.join(json_folder_path, "dummy_data.json"), "w") as f:
        #     json.dump(dummy_json, f)
        # print("Created a dummy JSON file for demonstration.")
        json_files = glob.glob(os.path.join(json_folder_path, "*.json"))

    total_papers = 0
    for file_path in json_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                paper_entries = content.get("data", [])
                for entry in paper_entries:
                    process_paper_entry(entry, g)
                    total_papers += 1
            except json.JSONDecodeError:
                print(
                    f"     Warning: Could not decode JSON from {file_path}. Skipping."
                )

    print(f"\nProcessed {total_papers} paper entries from {len(json_files)} files.")

    # Serialize the graph to a file
    g.serialize(destination=output_file, format="turtle")
    print(f"ABox graph saved to {output_file} with {len(g)} triples.")


if __name__ == "__main__":
    # IMPORTANT: Change this path to the folder where your JSON files are located.
    # The script will create this folder with a dummy file if it doesn't exist.
    json_data_folder = "data_json"
    output_rdf_file = "papers_abox.ttl"

    create_abox_from_json(json_data_folder, output_rdf_file)
