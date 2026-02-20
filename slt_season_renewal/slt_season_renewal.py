import argparse
import os

from fan360_ds_utils import (
    AttributeCreator,
    DatabaseManager,
    ModelLogger,
    CoreUtils,
    BOMQuerier,
    BOMReader,
)

from _version import __version__
from features import feature_engineering
from score import score
from train import train
from update_db_table import update_si_renewal_product


def main():
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", help="Either 'features', 'train', 'score', or 'update_db'"
    )
    args = parser.parse_args()
    if args.method not in {"features", "train", "score", "update_db"}:
        raise ValueError(
            "ERROR: must be one of 'features', 'train', 'score', or 'update_db"
        )

    environment = os.getenv("ENVIRONMENT")
    org_mnemonic = 'SLT'
    event_type = os.getenv("EVENT_TYPE",'Broadway')
    freq = os.getenv("FREQ","W")
    n_trials = None
    training_timeout = None

    model_logger = ModelLogger(
        model_name="slt-renewal-model",
        model_release=__version__,
        environment=environment,
    )
    core_utils = CoreUtils(
        environment=environment,
    )

    # feature engineering step
    if args.method == "features":
        bq = BOMQuerier(environment,
               register_boms=[
                   'ticket_history',
                   'ticket_purchaser',
                   'events',
                   'ticket_owner',
                   'mobile_comms',
                   'donor',
                   'touchpoints',
                   'fts_profile'
        ])
        br = BOMReader(environment=environment)
        feature_engineering(
            org_mnemonic, bq, model_logger, core_utils, br, freq, season_type_filter=event_type
        )
    # model training step
    elif args.method == "train":
        if event_type == "Concert":
            model_logger.set_experiment(f"datasci-SLT-renewal:{event_type}")
        else: 
            model_logger.set_experiment(f"datasci-SLT-renewal:{event_type}")
        train(model_logger, n_trials, training_timeout) 
    # scoring step
    elif args.method == "score":
        attr_creator = AttributeCreator(
            environment=environment,
        )
        if event_type == "Concert":
            model_logger.set_experiment(f"datasci-SLT-renewal:{event_type}")
        else: 
            model_logger.set_experiment(f"datasci-SLT-renewal:{event_type}")
        score(attr_creator, model_logger,season_type_filter=event_type,org_mnemonic=org_mnemonic)

    elif args.method == "update_db":
        db_manager = DatabaseManager(
            environment=environment,
            system="fan360db",
        )
        update_si_renewal_product(db_manager, model_logger, core_utils,season_type_filter=event_type,org_mnemonic=org_mnemonic)
        #TODO: need to figure out what all this update db stuff is.

if __name__ == "__main__":
    main()