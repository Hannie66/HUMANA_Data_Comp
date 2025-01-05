# Import necessary libraries
import pandas as pd
import numpy as np

# Set pandas display options for easier visualization of dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load datasets
ho_target_member = pd.read_csv('/Users/han/Desktop/Humana data comp/hold_out_clean/humana_mays_target_members_Holdout.csv')
ho_target_member_detail = pd.read_csv('/Users/han/Desktop/Humana data comp/hold_out_clean/humana_mays_target_member_details_Holdout.csv')
ho_target_member_condition = pd.read_csv('/Users/han/Desktop/Humana data comp/hold_out_clean/humana_mays_target_member_conditions_Holdout.csv')
ho_target_member_claim = pd.read_csv('/Users/han/Desktop/Humana data comp/hold_out_clean/humana_mays_target_member_visit_claims_Holdout.csv')

# Convert member details into a dataframe and inspect the structure
df_ho_target_member_detail = pd.DataFrame(ho_target_member_detail)
df_ho_target_member_detail.info()

# Convert member conditions into a dataframe and inspect the structure
df_ho_target_member_condition = pd.DataFrame(ho_target_member_condition)
df_ho_target_member_condition.info()

# Convert claims data into a dataframe and inspect the structure
df_ho_target_member_claim = pd.DataFrame(ho_target_member_claim)
df_ho_target_member_claim.info()

# Check for missing values in datasets
print(df_ho_target_member_condition.isnull().sum())
print(df_ho_target_member_claim.isnull().sum())
print(df_ho_target_member_detail.isnull().sum())

# Clean and preprocess the member details dataset
df_ho_target_member_detail = df_ho_target_member_detail.drop(
    ['race', 'plan_benefit_package_id', 'pbp_segment_id', 'county_of_residence', 'region'], axis=1
)
df_ho_target_member_detail['mco_contract_nbr'] = np.where(df_ho_target_member_detail['mco_contract_nbr'] == 'H5216', 1, 0)
df_ho_target_member_detail = df_ho_target_member_detail.dropna()

# Clean and preprocess the member conditions dataset
df_ho_target_member_condition = df_ho_target_member_condition.drop(
    ['chronicity', 'cond_desc', 'membership_year', 'cms_model_vers_cd'], axis=1
)
df_ho_target_member_condition['hcc_model_type'] = np.where(
    df_ho_target_member_condition['hcc_model_type'] == 'ESRD', 1, 0
)
df_ho_target_member_condition['cond_num'] = df_ho_target_member_condition.groupby('id')['cond_key'].transform('count')
df_ho_target_member_condition = df_ho_target_member_condition.drop(['cond_key'], axis=1)
df_ho_target_member_condition = df_ho_target_member_condition.drop_duplicates()

# Merge member details and conditions datasets
ho_condition_detail_merged = pd.merge(
    df_ho_target_member_detail, df_ho_target_member_condition, on='id', how='inner'
)

# Replace categorical values with binary encoding
ho_condition_detail_merged = ho_condition_detail_merged.replace({'Y': 1, 'N': 0, 'F': 1, 'M': 0})

# Preprocess claims data
df_ho_target_member_claim = df_ho_target_member_claim.drop(
    ['dos_year', 'ihwa', 'clm_unique_key', 'serv_date_skey'], axis=1
)
ho_claim_columns = [
    'pcp_visit', 'annual_wellness', 'humana_paf', 'preventative_visit', 
    'comp_physical_exam', 'fqhc_visit', 'telehealth', 'endocrinologist_visit', 
    'oncolologist_visit', 'radiologist_visit', 'podiatrist_visit', 
    'ophthalmologist_visit', 'optometrist_visit', 'physical_therapist_visit', 
    'cardiologist_visit', 'gastroenterologist_visit', 'orthopedist_visit', 
    'obgyn_visit', 'nephroloogist_visit', 'pulmonologist_visit', 
    'urgent_care_visit', 'er_visit'
]
df_ho_target_member_claim[ho_claim_columns] = df_ho_target_member_claim[ho_claim_columns].fillna(0).replace({'Y': 1}).astype(int)
df_ho_claim_grouped = df_ho_target_member_claim.groupby('id')[ho_claim_columns].max().reset_index()

# Merge claims data with the merged member details and conditions dataset
ho_claim_condition_detail = pd.merge(
    df_ho_claim_grouped, ho_condition_detail_merged, on='id', how='inner'
)

# Process and one-hot encode target member data
df_ho_target_member = pd.DataFrame(ho_target_member)
df_ho_target_member = df_ho_target_member.drop(['calendar_year', 'product_type', 'plan_category'], axis=1)
ho_target_claim_condition_detail = pd.merge(
    df_ho_target_member, ho_claim_condition_detail, on='id', how='inner'
)
ho_target_claim_condition_detail = pd.get_dummies(
    ho_target_claim_condition_detail, columns=['state_of_residence'], prefix='state'
)

# Save processed datasets
ho_target_claim_condition_detail.to_csv('ho_target_claim_condition_detail.csv', index=False)

# Additional data loading and merging (quality data, web activity, demographics, etc.)
ho_quality_data = pd.read_csv('/Users/han/Desktop/Humana data comp/hold_out_clean/ho_quality_data.csv')
df_ho_quality_data = pd.DataFrame(ho_quality_data)

# Merge all relevant datasets step-by-step
ho_conbine_data = pd.merge(ho_quality_data, ho_web_activity, on='id', how='inner')
ho_conbine_data = pd.merge(ho_conbine_data, ho_member_data, on='id', how='inner')
ho_conbine_data = pd.merge(ho_conbine_data, ho_demographics, on='id', how='inner')

# Save intermediate and final combined datasets
ho_conbine_data_1 = ho_conbine_data.drop(['lang_spoken_cd', 'rucc_category'], axis=1)
ho_conbine_data_1.to_csv('ho_conbine_data_1.csv', index=False)

ho_conbine_data_2 = pd.merge(ho_conbine_data_1, ho_PharmU, on='id', how='inner')
ho_conbine_data_2 = pd.merge(ho_conbine_data_2, ho_saleschannel, on='id', how='inner')
ho_conbine_data_2 = pd.merge(ho_conbine_data_2, ho_socialdet, on='id', how='inner')
ho_conbine_data_2.to_csv('ho_conbine_data_2.csv', index=False)

ho_conbine_data_3 = pd.merge(ho_conbine_data_2, ho_control_pt, on='id', how='inner')
ho_conbine_data_3 = pd.merge(ho_conbine_data_3, ho_cu, on='id', how='inner')
ho_conbine_data_3 = pd.merge(ho_conbine_data_3, ho_additional_features, on='id', how='inner')
ho_conbine_data_3 = pd.merge(ho_conbine_data_3, ho_target_claim_condition_detail, on='id', how='inner')

# Final cleanup and save
ho_conbine_data_3 = ho_conbine_data_3.drop(['channel'], axis=1)
ho_conbine_data_3.to_csv('ho_conbine_data_final.csv', index=False)
ho_final = ho_conbine_data_3.drop([
    'rwjf_homicides_rate', 'rwjf_high_school_pct', 'rwjf_violent_crime_rate',
    'rwjf_hiv_rate', 'rwjf_child_free_lunch_pct', 'rwjf_drinkwater_violate_ind',
    'rwjf_infant_mortality', 'rwjf_drug_overdose_deaths_rate',
    'rwjf_resident_seg_black_inx', 'rwjf_firearm_fatalities_rate',
    'rwjf_disconnect_youth_pct'
], axis=1)

ho_final_clean = ho_final.dropna()
ho_final_clean.to_csv('ho_final_clean.csv', index=False)
