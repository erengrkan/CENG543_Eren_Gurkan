import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib

def load_dbpedia_data(data_path="data"):
    """
    Downloads and loads the DBpedia-entity dataset.
    Returns corpus, queries, and qrels.
    """
    dataset = "dbpedia-entity"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    
    out_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent, data_path)
    data_path = util.download_and_unzip(url, out_dir)

    # Load data
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels
