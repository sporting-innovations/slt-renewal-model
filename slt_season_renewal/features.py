import json
import numpy as np
import pandas as pd
import datetime
#TODO: Add a logger

def derive_year(df,date_field):
    df[f"{date_field}_year"] = df[date_field].dt.year
    return df

def calculate_season_start_end_lookup(df_events,df_th):
    season_start_end_lookup = (
        df_events.loc[df_events['event_id'].isin(df_th['event_id'])]
        .pipe(derive_year,'start_datetime')
        .rename(columns={'start_datetime_year':'season'})
        .groupby('season').agg(current_season_start=('start_datetime','min'),current_season_end=('start_datetime','max'))
    )
    season_start_end_lookup.loc[:,'next_season_start']= season_start_end_lookup['current_season_start'].shift(-1)

    average_days_between_previous_season_end_and_current_season_start = (season_start_end_lookup['next_season_start'] - season_start_end_lookup['current_season_start']).dt.days.mean()

    season_start_end_lookup.loc[:,'next_season_start'] = season_start_end_lookup['next_season_start'].fillna(season_start_end_lookup['current_season_start']+datetime.timedelta(days=average_days_between_previous_season_end_and_current_season_start))
    season_start_end_lookup['1'] = 1
    return season_start_end_lookup

def explode_time_series_between_two_dates(df,start_date_field,end_date_field,freq='W'):
    def generate_freq_list(row):
        # Generates a list of Period objects for every freq in the interval
        return pd.period_range(
            start=row[start_date_field], 
            end=row[end_date_field], 
            freq=freq
        ).tolist()

    df['frequency_to_explode'] = df.apply(generate_freq_list, axis=1)
    df = df.explode('frequency_to_explode')
    df['start_of_freq'] = pd.to_datetime(df['frequency_to_explode'].dt.start_time)
    df['end_of_freq'] = pd.to_datetime(df['frequency_to_explode'].dt.end_time)
    mask = (
        (df['start_of_freq'].values >= pd.to_datetime(df[start_date_field]).values) & 
        (df['end_of_freq'].values <= pd.to_datetime(df[end_date_field]).values)
    )
    df = df.loc[mask].drop(columns=['frequency_to_explode'])

    return df

def filter_for_contains(df,field_to_search,contains_list):
    df = df.loc[df[field_to_search].isin(contains_list)]
    return df

def calculate_purchase_days(df,purchase_date='transaction_datetime',event_date='start_datetime'):
    df['purchase_days'] = (df[event_date]-df[purchase_date]).dt.days
    return df

def add_time_based_features(df,df_start_date,appendage,appendage_date_field,agg_dictionary,freq,append_level=['external_link_id'],fill=0,rename_columns=None):
    appendgage_date_freq_str = appendage_date_field+f"start_of_{freq}"
    appendage[appendgage_date_freq_str] = ( 
        pd.to_datetime(
            appendage[appendage_date_field].dt.to_period(freq=freq).dt.start_time
        )
    )
    appendage_grouped_by_freq = appendage.groupby(append_level+[appendgage_date_freq_str]).agg(agg_dictionary)
    appendage_agg_columns = list(appendage_grouped_by_freq.columns)
    appendage_grouped_by_freq = appendage_grouped_by_freq.reset_index()

    df = df.merge(appendage_grouped_by_freq, how='left',left_on=append_level+[df_start_date],right_on=append_level+[appendgage_date_freq_str])
    df[appendage_agg_columns] = df[appendage_agg_columns].fillna(fill)
    if rename_columns is not None:
        df = df.rename(columns=rename_columns)
    return df

def create_total_cumulative_and_season_cumulative_features(df,features):
    for each in features:
        df["cumulative_"+f"{each}"] = df.groupby(['external_link_id'])[each].transform('cumsum')
        df["cumulative_season_"+f"{each}"] = df.groupby(['external_link_id','season'])[each].transform('cumsum')
    return df

def feature_engineering(
    org_mnemonic, bq, model_logger, core_utils, br, freq, season_type_filter
):
    """
    Function to calculate final features for the renewal model

    Parameters
    ----------
    org_mnemonic: str
        FTS mnemonic for client
    bom_querier: BOMQuerier
        BOMQuerier object for reading data from s3 extracts
    model_logger: ModelLogger
        ModelLogger used to track model
    core_utils: CoreUtils
        CoreUtils used to load config data from toolbox
    bom_reader: BOMReader
        BOMReader object for reading profile attributes
    """

    #Pull Events
    df_events = bq.query(f"""
        select * from events
        where org_mnemonic = '{org_mnemonic}'
    """).to_df()
    df_events['start_datetime'] = pd.to_datetime(df_events['start_datetime'])

    event_cats = pd.read_csv('event_cats_filtered.csv')
    event_cats['NEW event_id'] = event_cats['NEW event_id'].fillna(0).astype(int).astype(str)
    df_events = df_events.merge(event_cats[['NEW event_id','updated_event_category']],how='left',left_on='event_id',right_on='NEW event_id')

    if season_type_filter == 'Concert':
        df_events = df_events.loc[df_events['updated_event_category']=='Concert']
    else:
        df_events = df_events.loc[df_events['updated_event_category']=='Broadway']

    #Pull Purchaser
    df_purchaser = bq.query(f"""
    select * from ticket_purchaser
    where org_mnemonic = '{org_mnemonic}'
    """).to_df()
    df_purchaser['transaction_datetime'] = pd.to_datetime(df_purchaser['transaction_datetime'])

    #Pull Ticket History
    df_th = bq.query(f"""
        select * from ticket_history
        where org_mnemonic = '{org_mnemonic}'
        and price_code = 'Subscription'
    """).to_df().map_external_links('produsa')
    df_th = df_th.pipe(derive_year,'transaction_datetime')
    df_th['ticket_id_customer_id'] = df_th['ticket_id']+"_"+df_th['customer_id']
    df_th['transaction_datetime'] = pd.to_datetime(df_th['transaction_datetime'])
    ticket_id_customer_id_list = df_th['ticket_id_customer_id'].unique()

    #Pull Scans (THIS IS NECESSARY BECAUSE SLT DOES NOT SHOW WHETHER A SCANNED TICKET IS A SEASON TICKET AS OF 2/19/2026)
    df_scans = bq.query(f"""
        select * from ticket_history
        where org_mnemonic = '{org_mnemonic}'
        and status = 'SCANNED'
    """).to_df().map_external_links('produsa')
    df_scans['ticket_id_customer_id'] = df_scans['ticket_id']+"_"+df_scans['customer_id']
    df_scans = df_scans.loc[df_scans['ticket_id_customer_id'].isin(ticket_id_customer_id_list)]
    df_scans['transaction_datetime'] = pd.to_datetime(df_scans['transaction_datetime'])

    #Pull Mobile Comms
    df_mobile_comms = bq.query(f"""
        select * from mobile_comms
        where org_mnemonic = '{org_mnemonic}'
    """).to_df()

    #Pull Donor
    df_donor = bq.query(f"""
        select * from donor
        where org_mnemonic = '{org_mnemonic}'
    """).to_df()

    #Pull Touchpoints
    df_touchpoints = bq.query(f"""
        select * from touchpoints
        where org_mnemonic = '{org_mnemonic}'
    """).to_df()

    #Pull FTS Profile
    df_profiles = br.read(
        "fts_profile",
        org_mnemonic=org_mnemonic,
        columns=[
            "external_link_id",
            "gender",
            "birth_date",
            "has_valid_first_name",
            "has_valid_last_name",
            "has_valid_phone",
            "has_valid_postal",
            "has_valid_email",
            "distance_to_venue_miles",
            "marital_status",
            "is_parent",
            "household_size",
            "household_income",
            "education_level",
        ],
    )
    df_profiles["has_dob"] = df_profiles["birth_date"].notna().astype("int")
    df_profiles[["has_valid_first_name",
            "has_valid_last_name",
            "has_valid_phone",
            "has_valid_postal",
            "has_valid_email",
            "is_parent",]] = df_profiles[["has_valid_first_name",
                                            "has_valid_last_name",
                                            "has_valid_phone",
                                            "has_valid_postal",
                                            "has_valid_email",
                                            "is_parent",]].astype(int)
    df_profiles[[
        'gender',
        'marital_status',
        'education_level',
    ]] =     df_profiles[[
        'gender',
        'marital_status',
        'education_level',
    ]].fillna("Unknown").astype('object')



    df_profiles["income_level"] = 'Unknown'
    df_profiles.loc[df_profiles["household_income"] >= 0, "income_level"] = "0-25k"
    df_profiles.loc[df_profiles["household_income"] >= 25000, "income_level"] = "25-50k"
    df_profiles.loc[df_profiles["household_income"] >= 50000, "income_level"] = "50-75k"
    df_profiles.loc[df_profiles["household_income"] >= 75000, "income_level"] = "75-100k"
    df_profiles.loc[df_profiles["household_income"] >= 100000, "income_level"] = "100-125k"
    df_profiles.loc[df_profiles["household_income"] >= 125000, "income_level"] = "125-150k"
    df_profiles.loc[df_profiles["household_income"] >= 150000, "income_level"] = "150-175k"
    df_profiles.loc[df_profiles["household_income"] >= 175000, "income_level"] = "175-200k"
    df_profiles.loc[df_profiles["household_income"] >= 200000, "income_level"] = "200k+"
    df_profiles = df_profiles.drop(["birth_date", "household_income"], axis=1)

    #Calculate Estimated Season Start and End Dates
    season_start_end_lookup = calculate_season_start_end_lookup(df_events=df_events,df_th=df_th)
    season_start_end_lookup = (
        season_start_end_lookup.pipe(explode_time_series_between_two_dates,'current_season_start','next_season_start','W')
    )

    #Use season_week_index_calculation to determine the weeks to rollup calculations at
    #Use season_week_index_prediction_week to enforce that we should only use aggregations from the prior weeks
    #when we are in the prediction week (e.g. Only use week 4's calculations, when predicting for week 5.)
    season_start_end_lookup['season_freq_index_calculation'] = season_start_end_lookup.groupby('season')['start_of_freq'].transform('cumcount')
    season_start_end_lookup['season_freq_index_prediction_freq'] = season_start_end_lookup['season_freq_index_calculation'] + 1
    season_start_end_lookup = season_start_end_lookup.reset_index()

    grouped_dfs_t = (
        df_purchaser
        .pipe(pd.merge,df_events,how='left',on='event_id',)
        .pipe(filter_for_contains,'ticket_id',df_th['ticket_id'].unique())
        .pipe(derive_year,'start_datetime')
        .pipe(calculate_purchase_days) #TODO: Double check this one. Make sure purchase days is the days from purchase to start of season.
    )

    #Not sure this replace is actually needed anymore, but won't hurt anything.
    grouped_dfs_t['updated_event_category'] = grouped_dfs_t['updated_event_category'].replace({
        'Movies':'Broadway',
        'Musicals':'Broadway',
        'Stand Up Comedy':'Broadway',
        'Special Guest':'Concert',
        'Chamber Music':'Concert',
        'Uncategorized':'Concert',
        'Visual Arts':'Other'
    })
    grouped_dfs_t['updated_event_category'].fillna('Broadway')

    #Enforce only looking at Concert or Broadway Events.
    if season_type_filter == 'Concert':
        grouped_dfs_t = grouped_dfs_t.loc[grouped_dfs_t['updated_event_category']=='Concert']
    else:
        grouped_dfs_t = grouped_dfs_t.loc[grouped_dfs_t['updated_event_category']=='Broadway']


    unique_dates = grouped_dfs_t['start_datetime_year'].unique()

    new_index = pd.MultiIndex.from_product([grouped_dfs_t['external_link_id'].unique(),unique_dates])
    grouped_dfs_2 = grouped_dfs_t.groupby(['external_link_id','start_datetime_year']).agg(
            price=('price','sum'),
            tickets_bought=('price','count'),
            purchase_days=('purchase_days','mean'), #TODO: I think this part is wrong
            date_purchased=('transaction_datetime','min')
    ).reindex(new_index).reset_index()
    grouped_dfs_2 = grouped_dfs_2.rename(columns={
        'level_0':'external_link_id',
        'level_1':'season'
    })

    grouped_dfs_2 = grouped_dfs_2.sort_values(by=['external_link_id','season'])
    grouped_dfs_2['did_buy_this_season'] = np.where(grouped_dfs_2['price'].notna(),1,0)
    #Adding some features here because it's easier. These are features that have to do with
    #the season ticket purchase itself, so doesn't need to be time sensitive, just external_link_id & season sensitive.
    grouped_dfs_2['rolling_seasons_bought'] = grouped_dfs_2.groupby(['external_link_id'])['did_buy_this_season'].transform('cumsum')
    grouped_dfs_2['rolling_price_sum'] = grouped_dfs_2.groupby(['external_link_id'])['price'].transform('cumsum')
    grouped_dfs_2['rolling_price_mean'] = grouped_dfs_2.groupby(['external_link_id'])['price'].transform(lambda x:x.expanding().mean())
    grouped_dfs_2['rolling_price_variance'] = grouped_dfs_2.groupby(['external_link_id'])['price'].transform(lambda x:x.expanding().var())
    grouped_dfs_2['last_years_price'] = grouped_dfs_2.groupby(['external_link_id'])['price'].transform('shift',1)
    grouped_dfs_2['price_yearly_delta'] = grouped_dfs_2['price']-grouped_dfs_2['last_years_price']
    grouped_dfs_2['tickets_bought_last_year'] = grouped_dfs_2.groupby('external_link_id')['tickets_bought'].transform('shift',1)
    grouped_dfs_2['tickets_bought_yearly_delta'] = grouped_dfs_2['tickets_bought'] - grouped_dfs_2['tickets_bought_last_year']
    grouped_dfs_2['last_years_purchase_days'] = grouped_dfs_2.groupby('external_link_id')['purchase_days'].transform('shift',1)
    grouped_dfs_2['purchase_days_yearly_delta'] = grouped_dfs_2['purchase_days'] - grouped_dfs_2['purchase_days']
    grouped_dfs_2['rolling_purchase_days_mean'] = grouped_dfs_2.groupby(['external_link_id'])['purchase_days'].transform(lambda x:x.expanding().mean())
    grouped_dfs_2['rolling_purchase_days_variance'] = grouped_dfs_2.groupby(['external_link_id'])['purchase_days'].transform(lambda x:x.expanding().var())
    grouped_dfs_2['average_ticket_price'] = grouped_dfs_2['price']/grouped_dfs_2['tickets_bought']
    grouped_dfs_2['bought_next_season'] = grouped_dfs_2.groupby(['external_link_id'])['did_buy_this_season'].transform('shift',periods=-1)
    grouped_dfs_2 = grouped_dfs_2.merge(season_start_end_lookup[['current_season_start','current_season_end','start_of_freq','end_of_freq','season','season_freq_index_calculation','season_freq_index_prediction_freq']],how='left')

    grouped_dfs_2.loc[:,'bought_next_season_this_week'] = np.where(
        (grouped_dfs_2['date_purchased'].values<=pd.to_datetime(grouped_dfs_2['end_of_freq']).values)
        &
        (grouped_dfs_2['date_purchased'].values>=pd.to_datetime(grouped_dfs_2['start_of_freq']).values),1,0)

    #Merge Profile Data
    grouped_dfs_2 = grouped_dfs_2.merge(df_profiles,how='left',on='external_link_id')
    #Need to add this fillna to account for external_link_ids that appear in ticket_history, but not in fts_profile.
    grouped_dfs_2[[
                'gender',
            'marital_status',
            'education_level',
        'income_level'
    ]] = grouped_dfs_2[['gender',
            'marital_status',
            'education_level',
                    'income_level']].fillna('Unknown')
    
    #Add Time Sensitive Donor Data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2,
                                appendage=df_donor,
                                df_start_date='start_of_freq',
                                appendage_date_field='received_date',
                                agg_dictionary={
                                    'payment_amount':'sum',
                                    'pledge_amount':'sum'
                                },
                                freq='W',fill=0)
    #Add Time Sensitive mobile comms data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_mobile_comms,
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='datetime',agg_dictionary={
                                                        'communication_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=pd.NA,
                                                    freq='W',
                                                    rename_columns={
                                                        'communication_id':'count_of_mobile_comms'
                                                    }
                                                    )
    #Add Time Sensitive mobile comms viewed data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_mobile_comms.loc[df_mobile_comms['status'].isin(
                                                        ["VIEWED","RECEIVED"])],
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='datetime',agg_dictionary={
                                                        'communication_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=pd.NA,
                                                    freq='W',
                                                    rename_columns={
                                                        'communication_id':'count_of_mobile_comms_viewed'
                                                    }
                                                    )
    #Add Time Sensitive mobile comms engaged data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_mobile_comms.loc[df_mobile_comms['status'].isin(
                                                        ["READ", "CLICKED","RESPONDED", "PLAYED", "LIKED"])],
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='datetime',agg_dictionary={
                                                        'communication_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=pd.NA,
                                                    freq='W',
                                                    rename_columns={
                                                        'communication_id':'count_of_mobile_comms_engaged'
                                                    }
                                                    )




    #Add Time Sensitive Scans Data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_scans,
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='transaction_datetime',agg_dictionary={
                                                        'ticket_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=0,
                                                    freq='W',
                                                    rename_columns={
                                                        'ticket_id':'count_of_scans'
                                                    }
                                                    )

    # Add Time Sensitive touchpoints data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_touchpoints,
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='create_datetime',agg_dictionary={
                                                        'note_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=0,
                                                    freq='W',
                                                    rename_columns={
                                                        'note_id':'num_touchpoints'
                                                    }
                                                    )
    # Add Time Sensitive connected touchpoints data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_touchpoints.loc[
                                                        df_touchpoints['touchpoint_type'].isin([
                                                            "EMAIL_CONNECTED",
                                                            "INBOUND_CALL",
                                                            "OUTBOUND_CALL_CONNECTED",
                                                            "APPOINTMENT",
                                                            "FACE_TO_FACE",
                                                        ])
                                                        ],
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='create_datetime',agg_dictionary={
                                                        'note_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=0,
                                                    freq='W',
                                                    rename_columns={
                                                        'note_id':'num_connected_touchpoints'
                                                    }
                                                    )
    # Add Time Sensitive connected touchpoints data
    grouped_dfs_2_with_tbf = add_time_based_features(grouped_dfs_2_with_tbf,
                                                    appendage=df_touchpoints.loc[
                                                        ~(df_touchpoints['touchpoint_type'].isin([
                                                            "EMAIL_CONNECTED",
                                                            "INBOUND_CALL",
                                                            "OUTBOUND_CALL_CONNECTED",
                                                            "APPOINTMENT",
                                                            "FACE_TO_FACE",
                                                        ]))
                                                        ],
                                                    df_start_date='start_of_freq',
                                                    appendage_date_field='create_datetime',agg_dictionary={
                                                        'note_id':'nunique'
                                                    },append_level=['external_link_id'],
                                                    fill=0,
                                                    freq='W',
                                                    rename_columns={
                                                        'note_id':'num_non_connected_touchpoints'
                                                    }
                                                    )

    #Calculcate Time Sensitive cumulative values across the years and within seasons.
    grouped_dfs_2_with_tbf = (
        grouped_dfs_2_with_tbf
        .pipe(create_total_cumulative_and_season_cumulative_features,[
            'payment_amount',
            'pledge_amount',
            'count_of_mobile_comms_viewed',
            'count_of_mobile_comms_engaged',
            'count_of_mobile_comms',
            'count_of_scans',
            'num_touchpoints',
            'num_connected_touchpoints',
            'num_non_connected_touchpoints'
        ])
    )

    #Figure out the freq index of when the season purchases are happening
    grouped_dfs_2_with_tbf['freq_of_purchase'] = np.where(
        ((grouped_dfs_2_with_tbf['date_purchased'].values >= pd.to_datetime(grouped_dfs_2_with_tbf['start_of_freq']).values)
            & (grouped_dfs_2_with_tbf['date_purchased'].values<=pd.to_datetime(grouped_dfs_2_with_tbf['end_of_freq']).values))
        |
        ((grouped_dfs_2_with_tbf['date_purchased']<=grouped_dfs_2_with_tbf['current_season_start'])
            &
            (grouped_dfs_2_with_tbf['season_freq_index_calculation']==0))
        ,1,0)


    t = grouped_dfs_2_with_tbf.loc[grouped_dfs_2_with_tbf['freq_of_purchase']==1,['external_link_id','season','season_freq_index_calculation']].sort_values(by=['external_link_id','season'])

    t['rolling_mean_freq_of_purchase'] = t.groupby('external_link_id')['season_freq_index_calculation'].transform(lambda x: x.expanding().mean())

    grouped_dfs_2_with_tbf = grouped_dfs_2_with_tbf.merge(t[['external_link_id','season','rolling_mean_freq_of_purchase']],how='left',on=['external_link_id','season'])
    grouped_dfs_2_with_tbf['freq_distance_from_average_purchase_freq'] = abs(grouped_dfs_2_with_tbf['season_freq_index_calculation']-grouped_dfs_2_with_tbf['rolling_mean_freq_of_purchase'])



    #TODO: These need to be reworked to use the rolling counts. It doesn't work when it is at the weekly level.
    #Calculate View Rates
    grouped_dfs_2_with_tbf['mobile_comms_view_rate'] = grouped_dfs_2_with_tbf['count_of_mobile_comms_viewed'] / grouped_dfs_2_with_tbf['count_of_mobile_comms']


    #Calculate engagement rates
    grouped_dfs_2_with_tbf['mobile_comms_engagement_rate'] = grouped_dfs_2_with_tbf['count_of_mobile_comms_engaged'] / grouped_dfs_2_with_tbf['count_of_mobile_comms']