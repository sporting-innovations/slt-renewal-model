import os
import pandas as pd  # type: ignore
from distutils.util import strtobool
import patsy
import datetime
from logzero import logger

def score(attr_creator, model_logger,season_type_filter,org_mnemonic):
    """
    Applies the model to a set of fans

    Parameters
    ----------
    attr_creator: AttributeCreator
        Attribute creator object for reading/writing to elasticsearch
    model_logger: ModelLogger
        ModelLogger used to load stored model

    """
    logger.info(f"Start Scoring")
    attribute_key = f"{season_type_filter.lower()}_renewal_risk_score" #keeping the name renewal_risk_score, but in this case it is likelihood to renew (bigger number= more likely)

    write_to_profile = bool(strtobool(os.getenv("WRITE_TO_PROFILE", "False")))

    logger.info(f"Scoring {model_logger.current_experiment.name}")

    # load model
    model = model_logger.load_model()
   
    # predict_proba() returns 0-indexed array, won't join with score_df that is not 0-indexed
    score_df = model_logger.load_experiment_artifact("score_df.parquet").reset_index(
        drop=True
    )

    logger.info(f"Filter Score to Current Week")
    season_freq_index_prediction_freq = score_df.loc[(score_df['start_of_freq']<=datetime.datetime.today()) & (score_df['end_of_freq']>=datetime.datetime.today()),'season_freq_index_calculation'].mean()-1
    score_df = score_df.loc[score_df['season_freq_index_calculation']==season_freq_index_prediction_freq]
    score_df['org_mnemonic'] = org_mnemonic

    score_df['target'] = score_df['bought_next_season']
    formula = """target ~  
            price +
            tickets_bought +
            purchase_days +
            rolling_seasons_bought +
            rolling_price_sum +
            rolling_price_mean +
            rolling_price_variance +
            last_years_price +
            price_yearly_delta +
            tickets_bought_last_year +
            tickets_bought_yearly_delta +
            last_years_purchase_days +
            purchase_days_yearly_delta +
            rolling_purchase_days_mean +
            rolling_purchase_days_variance +
            average_ticket_price +
            C(gender) +
            has_valid_first_name +
            has_valid_last_name +
            has_valid_phone +
            has_valid_postal +
            has_valid_email +
            distance_to_venue_miles +
            C(marital_status) +
            is_parent +
            household_size +
            C(education_level) +
            has_dob +
            C(income_level) +
            payment_amount +
            pledge_amount +
            count_of_mobile_comms +
            count_of_mobile_comms_viewed +
            count_of_mobile_comms_engaged +
            count_of_scans +
            num_touchpoints +
            num_connected_touchpoints +
            num_non_connected_touchpoints +
            cumulative_payment_amount +
            cumulative_season_payment_amount +
            cumulative_pledge_amount +
            cumulative_season_pledge_amount +
            cumulative_count_of_mobile_comms_viewed +
            cumulative_season_count_of_mobile_comms_viewed +
            cumulative_count_of_mobile_comms_engaged +
            cumulative_season_count_of_mobile_comms_engaged +
            cumulative_count_of_mobile_comms +
            cumulative_season_count_of_mobile_comms +
            cumulative_count_of_scans +
            cumulative_season_count_of_scans +
            cumulative_num_touchpoints +
            cumulative_season_num_touchpoints +
            cumulative_num_connected_touchpoints +
            cumulative_season_num_connected_touchpoints +
            cumulative_num_non_connected_touchpoints +
            cumulative_season_num_non_connected_touchpoints +
            rolling_mean_freq_of_purchase +
            freq_distance_from_average_purchase_freq 
        """


    _,X = patsy.dmatrices(formula, score_df, return_type = 'dataframe',NA_action=patsy.NAAction(NA_types=[]))



    # convert internal pandas column types to numpy compatible ones
    # score_df = cast_to_numpy(score_df.convert_dtypes(dtype_backend="numpy_nullable")) #I hope I don't need this, but leaving it here for now in case I do.

    # Prediction
    logger.info("Scoring")
    renewal_risk_score = model.predict_proba(X)[:,1]
    score_df[attribute_key] = renewal_risk_score
    score_df['scoring_date'] = pd.to_datetime("today")
    score_df.loc[score_df['bought_next_season']==1,attribute_key] = 1.0

    # if write to profile is true: log output results to s3 and write attributes to ES
    if write_to_profile:
        logger.info(f"Writing Score Attribute")
        # log results
        # keep just latest values for update_db_table
        model_logger.s3_manager.write_df(
            df=score_df,
            s3_path=f"{model_logger.artifact_uri_base}/{model_logger.current_experiment.name}/artifacts/output_df.parquet",
            partition_cols=["org_mnemonic"],
            partition_write_mode="overwrite",
        )
        # issue in duckdb > 1.0.0 prevents tableau extractor from including partition column (org_mnemonic) in datasource
        # workaround: include extra column with org_mnemonic/client
        # output_df["client"] = output_df["org_mnemonic"] #Don't think i need this anymore, but keeping around incase.
        # append results to all products file --> tableau datasource
        # model_logger.s3_manager.write_df(
        #     df=score_df.rename(columns={attribute_key: "renewal_risk_score"}),
        #     s3_path=f"{model_logger.artifact_uri_base}/output_df.parquet",
        #     partition_cols=["org_mnemonic"],
        #     partition_write_mode="append",
        # )

        # write current renewal scores to ES
        attr_creator.output_attr(
            attr_name=attribute_key,
            new_df=score_df[[attribute_key, "external_link_id", "org_mnemonic"]],
        )
    logger.info("Scoring complete")