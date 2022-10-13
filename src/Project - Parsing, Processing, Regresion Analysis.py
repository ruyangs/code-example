#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 13:58:03 2021

@author: ruyang, ziyue
"""

import os
import pandas as pd
import numpy as np
import re
import urllib.request
from bs4 import BeautifulSoup
import statsmodels.api as sm
import datetime
import geopandas

path = os.path.join(os.path.dirname(__file__), '../data')

########################### Parsing Data from Website ###########################


def get_csv(path, fname):
    csv = pd.read_csv(os.path.join(path, fname))
    return csv


def get_page(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, 'html.parser')
    return soup


def find_table(url, attrs):
    soup = get_page(url)
    table = soup.find('table', attrs=attrs)
    return table


def parse_table(table, tag):
    parsed_rows = []
    for row in table.find_all('tr'):
        td_tags = row.find_all(tag)
        parsed_rows.append([val.text.replace('\n', '') for val in td_tags])
    df = pd.DataFrame(parsed_rows)
    return df

############# Data Source 1: Stay-at-home Order Time Table #############

# Parsing the table on the web


def get_ordertime(url):
    table = find_table(url, None)
    df = parse_table(table, ['td'])
    df.columns = ['State', 'Effective Date',
                  'Duration or End Date', 'Resources']
    df = df.drop(index=[0]).drop(columns=['Resources']
                                 ).replace('\*', '', regex=True)
    return df

# Using Regex to clean the Stay-at-home Order Time Table


def regex_clean(column, character, regex):
    for i in df_ordertime[column]:
        if character in i:
            m = re.search(regex, i)
            df_ordertime[column] = df_ordertime[column].replace(i, m.group(0))
    return df_ordertime[column]


if not os.path.exists(os.path.join(path, 'df_ordertime.csv')):
    # Call the functions
    df_ordertime = get_ordertime(
        'https://www.littler.com/publication-press/publication/stay-top-stay-home-list-statewide')
    df_ordertime['State'] = regex_clean('State', '–', r'^\w+(?=)')
    df_ordertime['Effective Date'] = regex_clean(
        'Effective Date', ',', r'^([^,]*)')
    df_ordertime['Duration or End Date'] = regex_clean(
        'Duration or End Date', ',', r'^([^,]*)')

    def organize_ordertime():
        # modify special data
        df_ordertime.loc[[4, 16, 19, 28, 29, 34, 44],
                         'Duration or End Date'] = ['May 8', 'May 11', 'May 15', 'May 30', 'May 31', 'May 15', 'April 30']
    # change date format
        df_ordertime['start_date'] = df_ordertime['Effective Date'].apply(
            lambda s: '2020 '+s)
        df_ordertime['end_date'] = df_ordertime['Duration or End Date'].apply(
            lambda s: '2020 '+s)

        df_ordertime['start'] = df_ordertime['start_date'].apply(
            lambda i: datetime.datetime.strptime(i, '%Y %B %d'))
        df_ordertime['start'] = df_ordertime['start'].apply(
            lambda i: i.strftime('%m/%d/%Y'))
        df_ordertime['end'] = df_ordertime['end_date'].apply(
            lambda i: datetime.datetime.strptime(i, '%Y %B %d'))
        df_ordertime['end'] = df_ordertime['end'].apply(
            lambda i: i.strftime('%m/%d/%Y'))

        return df_ordertime

    df_ordertime = organize_ordertime()
    # write into csv
    df_ordertime.to_csv(os.path.join(path, 'df_ordertime.csv'))
else:
    df_ordertime = get_csv(path, 'df_ordertime.csv')

############# Data Source 2: Unemployment Rate by State #############

# Parse and merge all the states/regions and their links on the BLS into one dataframe
# Size of statelink:(53,2)


def get_statelink(url):
    parsed_statelink = []
    soup = get_page(url)
    for h4 in soup.select('h4'):
        state_name = h4.get_text(strip=True)
        parsed_statelink.append((state_name[:state_name.rfind('includes')],
                                 h4.a['href'].replace('\n', '')))
    df = pd.DataFrame(parsed_statelink, columns=['State', "Link"])
    df = df.drop(index=[11])
    return df


# Call the function
if not os.path.exists(os.path.join(path, 'df_statelink.csv')):
    df_statelink = get_statelink('https://www.bls.gov/eag/')

    # write into csv
    df_statelink.to_csv(os.path.join(path, 'df_statelink.csv'))
else:
    df_statelink = get_csv(path, 'df_statelink.csv')

# Get the unemployment page of each state


def get_statepage(state):
    link = state[1]['Link']
    html_page = urllib.request.urlopen(f'https://www.bls.gov{link}')
    soup = BeautifulSoup(html_page, 'html.parser')
    image = soup('div', {'align': 'center'})
    return image

# Parse the unemployment table of each state


def parse_unemployment(i):
    url_page = urllib.request.urlopen(i.a['href'])
    soup = BeautifulSoup(url_page, 'html.parser')
    table = soup.find('table', attrs={'id': 'table0'})
    df = parse_table(table, ['th', 'td'])
    df = df.rename(columns=df.iloc[0]).drop(index=[0])
    df.columns = df.columns.str.strip()
    df = df[(df['Year'] == '2020')]
    return df


if not os.path.exists(os.path.join(path, 'df_unemployment.csv')):
    # Parse and merge 53 tables into df_unemployment
    df_unemployment = pd.DataFrame()
    for state in df_statelink.iterrows():
        state_name = state[1]['State']
        image = get_statepage(state)
        for i in image:
            if '0000000000006' in i.img['alt']:
                df = parse_unemployment(i)
                df = df.set_index(['Period'])
                df = df.rename(columns={'unemployment rate': state_name})
                df_unemployment[state_name] = df[state_name]

    ############# Data Source 3: Unemployment Rate of total US #############

    # Get the unemployment page of the US
    soup = get_page('https://www.bls.gov/eag/eag.us.htm')
    state_name = 'United States'
    image2 = soup('div', {'align': 'center'})

    # Parse and merge the unemployment data of the US to df_unemployment
    for i in image2:
        if 'LNS14000000' in i.img['alt']:
            df = parse_unemployment(i).stack()
            df = df.drop(df.index[0])
            df = df.reset_index().drop(columns=['level_0'])
            df = df.set_index(['level_1']).rename_axis('Period')
            df_unemployment[state_name] = df

    # write into csv
    df_unemployment.to_csv(os.path.join(path, 'df_unemployment.csv'))
else:
    df_unemployment = get_csv(path, 'df_unemployment.csv')


############# Data Source 4: Employment Change of US by industry #############

# Get industry lists
# Size of industry lists: 12
soup = get_page('https://www.bls.gov/regions/southeast/alabama.htm#eag')

unparsed_industry = soup.select('p[class=sub0]')

industry_list = []
industry_list.append([val.text for val in unparsed_industry])
industry_list = industry_list[0][2:]
industry_list = [industry[:-3] for industry in industry_list]

if not os.path.exists(os.path.join(path, 'employees_by_industry.csv')):
    # Parse and Merge 53 X 12 tables into one big table: df_employees_industry
    # Notes: As it needs to parse 53 X 12 pages and tables, it would cost 5-10 mins to run
    df_employees_industry = pd.DataFrame()

    for state in df_statelink.iterrows():
        image = get_statepage(state)
        state_name = state[1]['State']
        df_per_state = pd.DataFrame()
        cnt = 0
        for idx in range(len(image)):
            if (idx <= 3 or idx % 2 == 0):
                continue

            i = image[idx]
            df = parse_unemployment(i).stack()
            df = df.drop(df.index[0])
            df_per_state[industry_list[cnt]] = df
            cnt += 1

        df_per_state = df_per_state.reset_index()
        df_per_state['State'] = state_name

        # append per state table
        df_employees_industry = df_employees_industry.append(df_per_state)

    # Index Adjustments
    df_employees_industry.loc[df_employees_industry.level_0 ==
                              'Dec', 'level_1'] = '2019Dec'
    df_employees_industry = df_employees_industry.drop(columns=['level_0'])
    df_employees_industry = df_employees_industry.rename(
        columns={'level_1': 'Month'})
    df_employees_industry = df_employees_industry.set_index(
        ['State', 'Month'], inplace=False)

    # write into csv
    df_employees_industry.to_csv(
        os.path.join(path, 'employees_by_industry.csv'))
else:
    df_employees_industry = get_csv(path, 'employees_by_industry.csv')

########################### Regression Analysis ###########################
# Regression model：unemployment rate = β0 + β1 10000cases + β2 GDP + β3 ordertime + β4 seasons
# Progress：
# 1、unifrom time unit
# 2、merge， by state & by date
# 3、add binary variables (ordertime/seasons)

############  Regression Datasets #############
df_mobility = get_csv(path, '2020_US_Region_Mobility_Report.csv')
df_mobility = df_mobility[df_mobility['sub_region_2'].isna()]
# write into csv
df_mobility.to_csv(os.path.join(path, 'df_mobility.csv'))

## 1cases ##


def get_df_cases():
    df_cases = get_csv(
        path, 'United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')
    # drop 2021 and other irrelevant columns
    df_cases = df_cases.loc[df_cases['submission_date'].str.contains("2020")]
    df_cases = df_cases.loc[:, ('submission_date', 'state', 'new_case')]
    return df_cases


df_cases = get_df_cases()
# write to csv
df_cases.to_csv(os.path.join(path, 'df_cases.csv'))


def get_df_10000cases():
    # calculate the rate
    df1 = df_cases.pivot(index='submission_date',
                         columns='state', values='new_case')
    df1['US'] = df1.T.sum()
    # fill nas
    df1 = df1.fillna(0)
    # set 10000 people as a unit
    df2 = df1/10000
    return df2


df_10000cases = get_df_10000cases()

## 2gdp ##
# drop 无关行列


def organize_gdp():
    df_gdp = get_csv(path, 'US-SQGDP1__ALL_AREAS_2005_2020.csv')
    df_gdp = df_gdp.loc[0:179, ]
    df_gdp = df_gdp.loc[df_gdp['LineCode'] == 1]
    df_gdp = df_gdp.loc[:, ('GeoName', '2020:Q1',
                            '2020:Q2', '2020:Q3', '2020:Q4')]

    return df_gdp


df_gdp = organize_gdp()
########## Regress us data ###########
# month dict


def get_regression_unemp():

    df_us_10000cases = df_10000cases['US'].reset_index()
    df_us = df_us_10000cases
    df_us = df_us.rename(columns={'US': '10000cases'})

    months = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
              '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
              '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
## organize US data ##
    df_us['month'] = 0

    for i in range(0, 345):
        df_us.loc[i, 'month'] = months[df_us.loc[i, 'submission_date'][0:2]]

    df_us['10000cases'] = df_us['10000cases'].replace(float('inf'), 0)

    # unemployment数据
    df_us_unemp = df_unemployment.loc[:, ['Period', 'United States']]
    df_us_unemp = df_us_unemp.rename(
        columns={'Period': 'month', 'United States': 'unemp'})

    # merge unemployment data
    df_us = pd.merge(df_us, df_us_unemp, on='month', how='outer')

    return df_us


df_us = get_regression_unemp()


def get_regression_gdp():
    # gdp data
    df_gdp = organize_gdp()
    df_gdp = df_gdp.set_index('GeoName')
    df_us_gdp = df_gdp.T

    df_us_gdp['United States']/92

    df_us['gdp'] = 0

    def quarter(x):
        if x in ['Jan', 'Feb', 'Mar']:
            return df_us_gdp.loc['2020:Q1', 'United States']/90
        elif x in ['Apr', 'May', 'Jun']:
            return df_us_gdp.loc['2020:Q2', 'United States']/91
        elif x in ['Jul', 'Aug', 'Sep']:
            return df_us_gdp.loc['2020:Q3', 'United States']/92
        else:
            return df_us_gdp.loc['2020:Q4', 'United States']/92

    df_us['gdp'] = df_us['month'].apply(lambda x: quarter(x))

    return df_us


df_us = get_regression_gdp()

# ordertime data


def get_regression_ordertime():
    df_us_ordertime = df_ordertime.loc[:, ['State', 'start', 'end']]
    df_us_ordertime.loc[46] = ['United States', max(
        df_us_ordertime['start']), max(df_us_ordertime['end'])]
    # add ordertime
    df_us['submission_date'] = df_us['submission_date'].apply(
        lambda i: datetime.datetime.strptime(i, '%m/%d/%Y'))
    df_us['submission_date'] = df_us['submission_date'].apply(
        lambda i: i.strftime('%m/%d/%Y'))

    start_date = df_us_ordertime[df_us_ordertime['State']
                                 == 'United States']['start'].values[0]
    end_date = df_us_ordertime[df_us_ordertime['State']
                               == 'United States']['end'].values[0]

    df_us['order'] = np.where((df_us.submission_date >= start_date) &
                              (df_us.submission_date <= end_date), 1, 0)

    return df_us


df_us = get_regression_ordertime()

# add seasons


def get_seasons(df_us):
    df_us['spring'] = np.where((df_us.submission_date >= '03/01/2020') &
                               (df_us.submission_date <= '05/30/2020'), 1, 0)
    df_us['summer'] = np.where((df_us.submission_date >= '06/01/2020') &
                               (df_us.submission_date <= '08/31/2020'), 1, 0)
    df_us['fall'] = np.where((df_us.submission_date >= '09/01/2020') &
                             (df_us.submission_date <= '11/30/2020'), 1, 0)
    df_us['winter'] = np.where((df_us.submission_date >= '12/01/2020') |
                               (df_us.submission_date <= '02/29/2020') &
                               (df_us.submission_date >= '01/01/2020'), 1, 0)

    return df_us


df_us = get_seasons(df_us)

# write into csv
df_us.to_csv(os.path.join(path, 'regression_us.csv'))

# regression model


def regression_model():
    X = df_us[["10000cases", "gdp", "order",
               "summer", "spring", "fall", "winter"]]
    y = df_us[["unemp"]]

    # add intercept
    X = sm.add_constant(X)

    # model
    model = sm.OLS(y.astype(float), X.astype(float)).fit()
    # result
    print(model.summary())


regression_model()

########################### Geodata Processing ###########################
################## Geoplot1 data ##################

## geodata ##


def get_geodata():
    geodata = os.path.join(path, 'tl_2017_us_state', 'tl_2017_us_state.shp')
    geodata = geopandas.read_file(geodata)
    df_statelink['State'] = df_statelink['State'].apply(
        lambda s: s.rstrip()).tolist()
    geodata = geodata[geodata['NAME'].isin(df_statelink['State'])]
    geodata = geodata.loc[:, ['NAME', 'geometry']]
    geodata = geodata[~geodata['NAME'].isin(['Alaska', 'Hawaii'])]

    return geodata


geodata = get_geodata()
# wirte into csv
geodata.to_file(os.path.join(path, 'geodata/geodata.shp'),
                driver='ESRI Shapefile',
                encoding='utf-8')

## merge case & mobility data into geodata ##


def get_df_cases():
    state_dict = {'US': 'United States', 'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'}

    df_cases = get_csv(
        path, 'United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')
    # drop 2021 and other irrelevant columns
    df_cases = df_cases.loc[df_cases['submission_date'].str.contains("2020")]
    df_cases = df_cases.loc[:, ('submission_date', 'state', 'new_case')]
    df_cases = df_cases[~ df_cases['state'].isin(
        ['GU', 'PR', 'NYC', 'MP', 'AS', 'PW', 'RMI', 'VI', 'FSM'])]
    df_cases = df_cases.reset_index()
    df_cases = df_cases.drop('index', axis=1)
    df_cases.groupby('submission_date').sum('new_cases')
    # uniform state name
    for i in range(0, 17595):
        df_cases.loc[i, 'state'] = state_dict[df_cases.loc[i, 'state']]
    df_cases_us = df_cases.groupby(
        'submission_date').sum('new_cases').reset_index()
    df_cases_us['state'] = 'United States'
    df_cases = df_cases.append(df_cases_us)
    df_cases = df_cases.rename(columns={'state': 'NAME'})

    return df_cases


df_cases = get_df_cases()
df_cases.to_csv(os.path.join(path, 'df_cases.csv'))


def get_geodata_cases():
    geodata_cases = pd.merge(geodata, df_cases, on='NAME', how='left')
    geodata_cases = geodata_cases[~geodata_cases['submission_date'].isna()]
    geodata_cases = geodata_cases.sort_values(
        'submission_date', axis=0, ascending=True)
    return geodata_cases


geodata_cases = get_geodata_cases()


def get_geo_mobility():
    # merge mobility data and get full data
    geodata_mobility = df_mobility[df_mobility['sub_region_1'].isin(geodata['NAME'].unique().tolist())].copy()
    # drop data
    a = []
    for i in range(len(geodata_mobility)):
        m = np.mean(geodata_mobility.iloc[i].to_list()[9:14])
        a.append(m)
    geodata_mobility['avg_mobility'] = a
    geodata_mobility = geodata_mobility.copy()
    geo_mobility = geodata_mobility.loc[:, [
        'sub_region_1', 'date', 'avg_mobility']]
    geo_mobility['date'] = geo_mobility['date'].apply(
        lambda i: datetime.datetime.strptime(i, '%Y-%m-%d'))
    geo_mobility['date'] = geo_mobility['date'].apply(
        lambda i: i.strftime('%m/%d/%Y'))
    geo_mobility = geo_mobility.rename(
        columns={'sub_region_1': 'NAME', 'date': 'submission_date'})

    return geo_mobility

geo_mobility = get_geo_mobility()
geo_mobility.to_csv(os.path.join(path, 'geo_mobility.csv'))
# merge

geodata_cases_mobility = pd.merge(geodata_cases, geo_mobility, on=[
                                  'NAME', 'submission_date'], how='left')


################## Geoplot2 data ##################
def get_geoplot2_data():
    df_employees_industry = get_csv(path, 'employees_by_industry.csv')
    # employees change rate by industry
    df_employees_industry = df_employees_industry.set_index(['State', 'Month'])
    df_employees_industry = df_employees_industry.pct_change().reset_index()
    df_employees_industry = df_employees_industry[df_employees_industry['Month'] != 'Jan']
    # merge into geodata
    geodata_industry = df_employees_industry.rename(columns={'State': 'NAME'})
    geo_industry = pd.merge(geodata, geodata_industry, on=['NAME'], how='left')

    return geo_industry


geo_industry = get_geoplot2_data()
