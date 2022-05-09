
import os
from .utils import graph_reader, feature_reader, parse_args, tab_printer
from .asne import ASNE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_asne(args, graph, features):
    """
    Fitting an ASNE model and saving the embedding.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :param features: Features in a dictionary.
    """
    model = ASNE(args, graph, features)
    model.train()
    model.save_embedding()