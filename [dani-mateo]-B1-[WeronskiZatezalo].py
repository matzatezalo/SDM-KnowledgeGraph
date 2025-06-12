import rdflib
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD


def create_knowledge_graph():
    """
    Creates and populates a knowledge graph based on the publications RDFS schema.
    """
    # --- 1. Setup: Namespaces and Graph ---
    # Define our custom namespace
    EX = Namespace("http://example.org/ontology/")

    # Create a new graph
    g = Graph()

    # Bind prefixes to the graph for cleaner serialization
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("ex", EX)

    # --- 2. TBox Definition (The Schema) ---
    # This section programmatically defines the classes and properties.

    print("Step 1: Defining TBox (Classes and Properties)...")

    # --- Class Definitions ---
    classes = {
        "Person": "Represents an individual, who can be an author or a reviewer.",
        "Paper": "A research paper written by one or more authors.",
        "Topic": "A keyword or a field of study related to a paper.",
        "Publication": "A superclass for any medium where papers are published.",
        "Volume": "A volume of a Journal.",
        "Edition": "The proceedings of a specific edition of an Event.",
        "Journal": "An academic journal that publishes papers in volumes.",
        "Event": "A superclass for conferences and workshops.",
        "Conference": "A well-established academic conference.",
        "Workshop": "An academic workshop, often for new trends.",
        "City": "The venue city for an event edition.",
        "Year": "The year a publication or event occurs.",
    }

    for name, comment in classes.items():
        class_uri = EX[name]
        g.add((class_uri, RDF.type, RDFS.Class))
        g.add((class_uri, RDFS.label, Literal(name)))
        g.add((class_uri, RDFS.comment, Literal(comment)))

    # --- Subclass Relationships ---
    g.add((EX.Volume, RDFS.subClassOf, EX.Publication))
    g.add((EX.Edition, RDFS.subClassOf, EX.Publication))
    g.add((EX.Conference, RDFS.subClassOf, EX.Event))
    g.add((EX.Workshop, RDFS.subClassOf, EX.Event))

    # --- Property Definitions ---
    # (domain, range, label, comment)
    # Use None if a property doesn't apply (e.g., for generic properties)
    properties = {
        # Object Properties
        "isAuthor": (
            EX.Person,
            EX.Paper,
            "is author",
            "Connects an author to a paper they have written.",
        ),
        "isCorrespondingAuthor": (
            EX.Person,
            EX.Paper,
            "is corresponding author",
            "Connects a person to a paper as a corresponding author (cannot enforce singlurarity)",
        ),
        "isReviewer": (
            EX.Person,
            EX.Paper,
            "is reviewer",
            "Connects a person to a paper they review.",
        ),
        "publishedIn": (
            EX.Paper,
            EX.Publication,
            "published in",
            "Indicates that a paper is published in a specific volume or edition.",
        ),
        "belongsTo": (
            None,
            None,
            "belongs to",
            "A property to link a publication part to its whole.",
        ),
        "volumeBelongsToJournal": (
            EX.Volume,
            EX.Journal,
            "volume belongs to journal",
            "Links a volume to its journal.",
        ),
        "editionBelongsToEvent": (
            EX.Edition,
            EX.Event,
            "edition belongs to event",
            "Links an edition to its event.",
        ),
        "cites": (EX.Paper, EX.Paper, "cites", "A paper citing another paper."),
        "isAbout": (
            EX.Paper,
            EX.Topic,
            "is about",
            "Connects a paper to its topics/keywords.",
        ),
        "hostedIn": (
            EX.Edition,
            EX.City,
            "hosted in",
            "Specifies the city where an event edition is held.",
        ),
        "publishedInYear": (
            EX.Publication,
            EX.Year,
            "published in year",
            "The year a volume or edition was published.",
        ),
        # ============================
        # Datatype Properties
        "hasAbstract": (
            EX.Paper,
            XSD.string,
            "has abstract",
            "The summary of the paper.",
        ),
        "hasName": (None, XSD.string, "has name", "A generic name property."),
        "hasId": (EX.Volume, XSD.string, "has id", "The identifier of a volume."),
        "hasField": (
            EX.Topic,
            XSD.string,
            "has field",
            "The name of the topic or field.",
        ),
        "hasEventType": (
            EX.Event,
            XSD.string,
            "has event type",
            "Type of event (e.g., 'Conference').",
        ),
        "yearValue": (EX.Year, XSD.gYear, "year value", "The actual year value."),
    }

    for name, (domain, range, label, comment) in properties.items():
        prop_uri = EX[name]
        g.add((prop_uri, RDF.type, RDF.Property))
        if domain:
            g.add((prop_uri, RDFS.domain, domain))
        if range:
            g.add((prop_uri, RDFS.range, range))
        g.add((prop_uri, RDFS.label, Literal(label)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))

    # --- Subproperty Relationships ---
    g.add((EX.isCorrespondingAuthor, RDFS.subPropertyOf, EX.isAuthor))
    g.add((EX.volumeBelongsToJournal, RDFS.subPropertyOf, EX.belongsTo))
    g.add((EX.editionBelongsToEvent, RDFS.subPropertyOf, EX.belongsTo))

    print("TBox definition complete.")

    # --- 4. Serialize the Graph ---
    # Save the graph to a file in Turtle format
    output_file = "knowledge_graph.ttl"
    g.serialize(destination=output_file, format="turtle")

    print(f"\nKnowledge graph saved to {output_file}")

    # Optional: Print the graph to the console
    print("\n--- Graph Content (Turtle) ---")
    print(g.serialize(format="turtle"))

    return g


if __name__ == "__main__":
    knowledge_graph = create_knowledge_graph()
    print(f"\nGraph created successfully with {len(knowledge_graph)} triples.")
