#%% Necessary libraries and data
import pandas as pd
import numpy as np
import os
from pandasgui import show
import pandas_profiling as ppf
import functools
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from tqdm import tqdm

def my_read_csv(file_name, **kw):
    if '.csv' not in file_name:
        file_name = '%s.csv'%file_name
    if all([i not in file_name for i in ['\\','/']]):
        file_name = r'.\%s'%file_name
    return pd.read_csv(
        filepath_or_buffer=file_name,
        encoding='gbk', dtype=object, 
        keep_default_na=False, 
        na_values=['-1.#IND', '1.#QNAN', '1.#IND',
                   '-1.#QNAN', '#N/A N/A','#N/A', 
                   'N/A', '#NA', 'NULL', 'NaN', 
                   '-NaN', 'nan', '-nan', ''], 
        **kw)

def my_save_csv(df, path, **kw):
    if '.csv' not in path:
        path = '%s.csv'%path
    if all([i not in path for i in ['\\','/']]):
        path = r'.\%s'%path
    df.to_csv(path, index=False, encoding='gbk', **kw)

def col2date(df,lst,errors_='coerce'):
    for date_col in lst:
        df[date_col] = pd.to_datetime(df[date_col], errors=errors_)\
                       .dt.normalize()

def col2num(df,lst,errors_='coerce'):
    for num_col in lst:
        df[num_col] = pd.to_numeric(df[num_col], errors=errors_)

# change work direction
os.chdir(r'D:\作业文件\研究生\【1】文章\急性HIV18-21英文\abnormality_rate_analysis_script_and_data')


#%% obtaining first positive HIV screening test
all_test_results = my_read_csv('all_test_results_without_time_of_first_postive_screening_test')
hivPosLuminResults = all_test_results\
    .query("test_name.isin(['HIVCOMPT', 'HIVCOM', 'HIVDUO'])")\
    .copy()
hivPosLuminResults.test_name = hivPosLuminResults.test_name\
    .str.replace('HIVCOMPT','HIVCOM')
hivPosLuminResults = hivPosLuminResults\
    .query('~abnormal_ornot.str.contains("阴")')
hivPosLuminResults['audit_time'] = hivPosLuminResults.apply(
    lambda se: se['audit_time'] \
        if se['audit_time'] is not np.nan \
        else se['test_time'], 
    axis=1)
col2date(hivPosLuminResults,['audit_time'])
hivPosLuminResults = hivPosLuminResults\
    .sort_values(by=['ID','audit_time'])\
    .drop_duplicates(subset=['ID'], keep='first')
# my_save_csv(hivPosLuminResults, r'.\first_postive_screening_test.csv')

#%% 
'''
calculating the time difference between each record and the
first positive HIV screening test of corresponding patients
'''

all_test_results1 = my_read_csv(r'.\all_test_results_without_time_of_first_postive_screening_test.csv')
hivPosLuminResults = my_read_csv(r'.\first_postive_screening_test.csv')
pos1stDict = hivPosLuminResults.set_index(['ID'])['audit_time'].to_dict()
all_test_results1.insert(
    8,'time_1st_pos',
    all_test_results1.ID.map(pos1stDict))
col2date(all_test_results1,['audit_time', 'time_1st_pos'])

all_test_results1.insert(
    9, 'time_to_1st_pos',
    all_test_results1.apply(
        lambda se: (se['audit_time'] - se['time_1st_pos']).days, 
        axis=1)
    )

all_test_results1.insert(
    10, 'abs_time_to_1st_pos',
    all_test_results1.time_to_1st_pos.map(abs)
    )

all_test_results1 = all_test_results1.rename(columns={'ID':'ID'})
# my_save_csv(all_test_results1, r'.\all_test_results.csv')

#%% 

'''
a portion of panel include tests with same name,
which are not in fact equivalent, for example, 
both blood routine and Stool routine include a test 
called RBC(red blood cell), but they are in fact different tests

Addressing this problem requires to distinguish the test name 
by the panel name, however, when multiple panel are test within 
one sample, these panel would be linked by a "+" sign and palced 
in the panel name field of the sample's record. for example, 
there is a panel name like "Biochemical routine 1 + Biochemical routine 4"

these record are split using the following code
'''
# read all test records
all_test_results = my_read_csv(r'.\all_test_results.csv')

# dictionary of panel and corresponding test
test_panel_df = my_read_csv('./test_name_panel_name.csv')

def panel2list(se):
    panel_list = list(se)
    panel_list.sort(key = lambda i:len(i),reverse=True) 
    return panel_list

test_panel_df = test_panel_df\
    .groupby(['test_name'])['panel_name']\
    .apply(panel2list)\
    .reset_index()

test2panel = test_panel_df\
    .set_index(['test_name'])['panel_name']\
    .to_dict()

# e.g. remove_substr(['abc', 'df', 'ab']) return ['abc', 'df']
def remove_substr(lst):
    lst = list(set(lst))
    for i in lst:
        for j in lst:
            if len(j)<len(i) and j in i:
                lst.remove(j)
            else:
                pass
    return lst

# adjust panel name (remove the irrelevent panel name that connected by "+")
def adjust_panel_name(se):

    bool_lst = list(map(
            lambda x: x in se['panel_name'], 
            test2panel[se['test_name']]))\
        if se['test_name'] in test2panel\
        else [False]
    
    if any(bool_lst):
        substr_lst = list(filter(
            lambda x: x in se['panel_name'], 
            test2panel[se['test_name']]))
        substr_lst = remove_substr(substr_lst)
        substr_lst.sort()
        return '+'.join(substr_lst)

    elif se['test_name'] in se['panel_name']:
        return se['test_name']
    else:
        return se['panel_name']

'''
a portion of panel include same test, for example, 
"biochemical routine 2" is a subpanel of "biochemical routine 1", 
and they both contain the test "blood glucose"

Therefore, "biochemical routine 1" and "biochemical routine 2" 
should be both categorised as blood biochemistry.

This categorisation process is accomplished by the following functions
'''

panel_type_df = my_read_csv('./panel_category.csv')
panel_type = panel_type_df.set_index('panel_name')['panel_category'].to_dict()


# Categorizing panel
def compute_panel_type(se):
    if '+' in se['panel_name1']:
        adjusted_panel = se['panel_name1'].split('+')
        adjusted_panel.append(se['panel_name1'])
    else:
        adjusted_panel = [se['panel_name1']]
    panel_type_se = pd.Series(adjusted_panel, dtype=object).map(panel_type)
    panel_type_se = panel_type_se[~panel_type_se.isna()]
    panel_type_se = panel_type_se.drop_duplicates()
    panel_type1 = '+'.join(panel_type_se) if len(panel_type_se)>0 else np.nan
    return panel_type1

tqdm.pandas(desc="my bar")
all_test_results.insert(
    13,'panel_name1', 
    all_test_results.progress_apply(adjust_panel_name, axis=1))

all_test_results.insert(
    14,'panel_category', 
    all_test_results.progress_apply(compute_panel_type, axis=1))

my_save_csv(all_test_results, 'all_test_results_adjusted_panel_name')


#%% calculating abnormality rate
all_test_results = my_read_csv('all_test_results_adjusted_panel_name')
col2num(all_test_results, ['time_to_1st_pos', 'abs_time_to_1st_pos'])
col2date(all_test_results, ['test_time', 'audit_time', 'time_1st_pos'])


'''
a record would only be reserved if the absolute value of the time difference
were less or equal to 3
'''

abs_less_than_3 = all_test_results.query('abs_time_to_1st_pos<=3')

# Keeping only the first records of the same test for a same patient
abs_less_than_3 = abs_less_than_3\
    .sort_values(by=['ID', 'test_name', 'panel_category', 'audit_time'])\
    .drop_duplicates(subset=['ID', 'test_name', 'panel_category'],keep='first')\
    .copy()

# Extract positive samples
abs_less_than_3_p = abs_less_than_3\
    .query('~abnormal_ornot.isna()')\
    .query('~abnormal_ornot.str.contains("阴")')

# total patient number of each test
abs_less_than_3_gr = abs_less_than_3\
    .groupby(['test_name','panel_category','name'])\
    .size().reset_index()\
    .groupby(['test_name','panel_category'])\
    .size().reset_index()\
    .rename(columns={0:'total_number'})

# positive patient number of each test
abs_less_than_3_p_gr = abs_less_than_3_p\
    .groupby(['test_name','panel_category','name'])\
    .size().reset_index()\
    .groupby(['test_name','panel_category'])\
    .size().reset_index()\
    .rename(columns={0:'positive_number'})

# Descriptive statistics for each test
col2num(abs_less_than_3, ['test_result'])
abs_less_than_3_gr_describe = abs_less_than_3\
    .loc[:,['test_name','panel_category','test_result']]\
    .groupby(['test_name','panel_category'])\
    .describe()
abs_less_than_3_gr_describe.columns = abs_less_than_3_gr_describe.columns.droplevel(0)
abs_less_than_3_gr_describe = abs_less_than_3_gr_describe\
    .reset_index().drop(columns=['count'])

# merge total patient number and positive patient number
num_of_test_and_positive_test = pd.merge(
    abs_less_than_3_gr, 
    abs_less_than_3_p_gr, 
    on=['test_name','panel_category'], 
    how='outer')

'''
the positive number of tests be calculated with the above code
if the number was 0, and were set as 0 manually
'''
num_of_test_and_positive_test = num_of_test_and_positive_test\
    .fillna({'positive_number':0})

# Add descriptive statistics for each test
num_of_test_and_positive_test = pd.merge(
    abs_less_than_3_gr_describe, 
    num_of_test_and_positive_test, 
    on=['test_name','panel_category'], 
    how='outer')

# Calculating abnormality rate
num_of_test_and_positive_test['abnormality_rate'] = \
    num_of_test_and_positive_test['positive_number']\
    /num_of_test_and_positive_test['total_number']

# abandoning tests of which the abnormal patient number ≤ 5
num_of_test_and_positive_test = num_of_test_and_positive_test\
    .query('positive_number>5')\
    .sort_values('abnormality_rate', ascending=False)

# Changing the format of the abnormality rate from decimal to percentage
num_of_test_and_positive_test.abnormality_rate = num_of_test_and_positive_test.abnormality_rate\
    .map(lambda x: f'{x:.0%}')

my_save_csv(num_of_test_and_positive_test, 'tests_with_high_abnormality_rate')

#%% Reframing the table to make it easier to read
high_abnormal_rate_describe = my_read_csv('tests_with_high_abnormality_rate')
col2num(
    high_abnormal_rate_describe, 
    ['mean', 'std', 'min', '25%', '50%', '75%', 'max'])

high_abnormal_rate_describe['abnormality_rate(abnormal_numbers/total_numbers)'] = \
    high_abnormal_rate_describe.apply(
        lambda se: f"{se['abnormality_rate']}({se['positive_number']}/{se['total_number']})", 
        axis=1)

high_abnormal_rate_describe['mean ± std'] = \
    high_abnormal_rate_describe.apply(
        lambda se: \
            f"{se['mean']:.2f} ± {se['std']:.2f}" \
            if not pd.isnull(se['mean']) \
            else np.nan, 
        axis=1)

high_abnormal_rate_describe['median (IQR)'] = \
    high_abnormal_rate_describe.apply(
        lambda se: \
            f"{se['50%']:.2f} ({se['25%']:.2f},{se['75%']:.2f})" \
            if not pd.isnull(se['mean']) \
            else np.nan, 
        axis=1)

high_abnormal_rate_describe['min-max'] = \
    high_abnormal_rate_describe.apply(
        lambda se: f"{se['min']:.2f}-{se['max']:.2f}" 
            if not pd.isnull(se['mean']) 
            else np.nan, 
        axis=1)

high_abnormal_rate_describe = high_abnormal_rate_describe\
    .loc[:,['test_name', 'panel_category', 
            'abnormality_rate(abnormal_numbers/total_numbers)', 
            'min-max', 'median (IQR)', 'abnormality_rate', 
            'positive_number', 'total_number']]

type_test_refrange_df = my_read_csv('./test_name_panel_category_reference_range.csv')

high_abnormal_rate_describe = pd.merge(
    high_abnormal_rate_describe, 
    type_test_refrange_df, 
    left_on=['test_name', 'panel_category'], 
    right_on=['test_name', 'panel_category'], 
    how='left')

high_abnormal_rate_describe['abnormality_rate'] = \
    high_abnormal_rate_describe['abnormality_rate']\
        .map(lambda x:x[:-1])

col2num(high_abnormal_rate_describe,
    ['abnormality_rate', 'positive_number', 'total_number'])

# Excluding tests for HIV diagnosis
high_abnormal_rate_describe = high_abnormal_rate_describe\
    .query("~test_name.isin(['HIVDUO', 'HIVCOM', 'HIVP24Ag', \
                             'HIVAb', '高精度HIV-I病毒载量'])")

# my_save_csv(high_abnormal_rate_describe,'tests_with_high_abnormality_rate_exhibition version')
