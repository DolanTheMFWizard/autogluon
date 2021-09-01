import logging

from autogluon.core.dataset import TabularDataset
from autogluon.core.features.feature_metadata import FeatureMetadata

try:
    from .version import __version__
except ImportError:
    pass

from .predictor import TabularPredictor
from .bad_pseudo import fit_pseudo_end_to_end

logging.basicConfig(format='%(message)s')  # just print message in logs
