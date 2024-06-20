import pandas as pd
import numpy as np
import pickle
import os
import datetime
import patsy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# First, we'll clean up Portland Maps API data and save only detached, single family homes
# 

#get data
zillow = pd.read_pickle('zillow.pkl')
df = pd.read_pickle('home_results.pkl')


#only need certain zillow features
zillow = zillow[['roof', 'exteriormaterial', 'parkingtype', 'heatingsystem', 'coolingsystem',  'architecture', 'homedescription', 'elementaryschool']]
#merge data sets
df = pd.concat([df, zillow], axis = 1)


#delete unnecessary columns - we have most of these already from the portland maps data set
df = df[['address', 'sale_price', 'description','sale_date','property_tax_2016','lot_size', 
         'square_feet', 'main_sqft', 'zillow_bedrooms', 'plumbing', 'year_built', 'type', 
         'neighborhood',  'zillow_id', 'zillow_finished_sqft', 'zip_code', 'elementaryschool', 'segments', 'is_condo', 
        'roof', 'exteriormaterial', 'parkingtype', 'heatingsystem', 'coolingsystem', 
         'architecture','homedescription']]


#consider single family homes only
df = df[df['description']=='SINGLE FAMILY RESIDENTIAL']
del df['description']

#keep only detached homes
keep_list = ['1 STY', '1 STY W/BSMT', '1 STY W/ATTIC & BSMT', '2 OR MORE STY', '2 OR MORE STY W/BSMT', '1 STY W/ATTIC', '2 OR MORE STY W/ATTIC & BSMT', '2 OR MORE STY W/ATTIC', '']
df = df[df['type'].isin(keep_list)]


#get rid of one missing row
df = df[df['plumbing'].notnull()]

#convert to correct types
df['year_built'] = df['year_built'].astype(int)
df['lot_size']=df['lot_size'].astype(int)

df['main_sqft'] = df['main_sqft'].astype(int)
df['segments'] = df['segments'].astype(int)
df['is_condo'] = df['is_condo'].astype(int)
df['zip_code'] = df['zip_code'].astype(object)
df['zillow_id'] = df['zillow_id'].astype(object)

df['property_tax_2016'] = df['property_tax_2016'].str.replace('$', '')
df['property_tax_2016'] = df['property_tax_2016'].str.replace(',', '')
df['property_tax_2016'] = df['property_tax_2016'].astype(float)


#create a flag for missing on zillow
df['on_zillow'] = 0
df.loc[df['zillow_id'].notnull(),'on_zillow'] = 1
del df['zillow_id']


#fill in missing info
df['neighborhood'] = df['neighborhood'].fillna('missing')

bedroom_mean = int(np.mean([float(x) for x in df[df['zillow_bedrooms'].notnull()].zillow_bedrooms.values]))
df['zillow_bedrooms'].fillna(bedroom_mean, inplace = True)
df['zillow_bedrooms'] = df['zillow_bedrooms'].astype(int)

finished_sqft_mean = int(np.mean([float(x) for x in df[df['zillow_finished_sqft'].notnull()].zillow_finished_sqft.values]))
df['zillow_finished_sqft'].fillna(bedroom_mean, inplace = True)
df['zillow_finished_sqft'] = df['zillow_finished_sqft'].astype(int)


df = df[df.index.values != 1] #picture1 is corrupted so get rid of that row


#condense to 60 neighborhoods
df['neighborhood'].replace('ALAMEDA/BEAUMONT-WILSHIRE', 'ALAMEDA', inplace = True)
df['neighborhood'].replace('HEALY HEIGHTS/SOUTHWEST HILLS RESIDENTIAL LEAGUE', 'SOUTHWEST HILLS RESIDENTIAL LEAGUE', inplace = True)
df['neighborhood'].replace('SUNDERLAND ASSOCIATION OF NEIGHBORS', 'EAST COLUMBIA', inplace = True)
df['neighborhood'].replace('WOODLAND PARK', 'PARKROSE', inplace = True)
df['neighborhood'].replace('ARDENWALD-JOHNSON CREEK/WOODSTOCK', 'ARDENWALD-JOHNSON CREEK', inplace = True)
df['neighborhood'].replace('MC UNCLAIMED #5', 'GRANT PARK/HOLLYWOOD', inplace = True)
df['neighborhood'].replace('LLOYD DISTRICT COMMUNITY ASSOCIATION', "SULLIVAN'S GULCH", inplace = True)
df['neighborhood'].replace('BRIDLEMILE/SOUTHWEST HILLS RESIDENTIAL LEAGUE', "BRIDLEMILE", inplace = True)
df['neighborhood'].replace('HOLLYWOOD', "GRANT PARK/HOLLYWOOD", inplace = True)
df['neighborhood'].replace('ALAMEDA/IRVINGTON COMMUNITY ASSN.', 'ALAMEDA', inplace = True)
df['neighborhood'].replace('ARDENWALD-JOHNSON CREEK', 'BRENTWOOD-DARLINGTON', inplace = True)
df['neighborhood'].replace('SYLVAN-HIGHLANDS/SOUTHWEST HILLS RESIDENTIAL LEAGUE', 'SYLVAN-HIGHLANDS', inplace = True)
df['neighborhood'].replace('CENTENNIAL COMMUNITY ASSN./PLEASANT VALLEY', 'CENTENNIAL COMMUNITY ASSOCIATION', inplace = True)
df['neighborhood'].replace('SABIN COMMUNITY ASSN./IRVINGTON COMMUNITY ASSN.', 'SABIN COMMUNITY ASSOCIATION', inplace = True)
df['neighborhood'].replace('GLENFAIR', 'CENTENNIAL COMMUNITY ASSOCIATION', inplace = True)
df['neighborhood'].replace('GOOSE HOLLOW FOOTHILLS LEAGUE/SOUTHWEST HILLS RESIDENTIAL LEAGUE', 'GOOSE HOLLOW FOOTHILLS LEAGUE', inplace = True)
df['neighborhood'].replace('MAYWOOD PARK', 'PARKROSE HEIGHTS', inplace = True)
df['neighborhood'].replace('ARLINGTON HEIGHTS', 'HILLSIDE', inplace = True)
df['neighborhood'].replace('BRIDGETON', 'EAST COLUMBIA', inplace = True)
df['neighborhood'].replace('GRANT PARK/HOLLYWOOD', 'GRANT PARK', inplace = True)
df['neighborhood'].replace('FOREST PARK', 'HILLSIDE', inplace = True)
df['neighborhood'].replace('HILLSIDE/NORTHWEST DISTRICT ASSN.', 'HILLSIDE', inplace = True)
df['neighborhood'].replace('OLD TOWN/CHINATOWN COMMUNITY ASSOCIATION', 'PORTLAND DOWNTOWN', inplace = True)
df['neighborhood'].replace('HOMESTEAD', 'SOUTHWEST HILLS RESIDENTIAL LEAGUE', inplace = True)
df['neighborhood'].replace('HOMESTEAD', 'SOUTHWEST HILLS RESIDENTIAL LEAGUE', inplace = True)
df['neighborhood'].replace('LINNTON', 'ST. JOHNS', inplace = True)
df['neighborhood'].replace('PLEASANT VALLEY/POWELLHURST-GILBERT', 'POWELLHURST-GILBERT', inplace = True)
df['neighborhood'].replace('KERNS', 'BUCKMAN COMMUNITY ASSOCIATION', inplace = True)
df['neighborhood'].replace('SYLVAN-HIGHLANDS', 'HILLSIDE', inplace = True)
df['neighborhood'].replace('REED', 'CRESTON-KENILWORTH', inplace = True)
df['neighborhood'].replace('SUMNER ASSOCIATION OF NEIGHBORS', 'EAST COLUMBIA', inplace = True)
df['neighborhood'].replace('LENTS/POWELLHURST-GILBERT', 'POWELLHURST-GILBERT', inplace = True)
df['neighborhood'].replace('VERNON', 'BOISE', inplace = True)
df['neighborhood'].replace('HUMBOLDT', 'BOISE', inplace = True)
df['neighborhood'].replace("ELIOT", 'BOISE', inplace = True)
df['neighborhood'].replace("SULLIVAN'S GULCH", 'IRVINGTON COMMUNITY ASSOCIATION', inplace = True)
df['neighborhood'].replace("SULLIVAN'S GULCH", 'IRVINGTON COMMUNITY ASSOCIATION', inplace = True)
df['neighborhood'].replace("RUSSELL", 'PARKROSE', inplace = True)

#replace missing neighborhoods with their zip code
for i in df.index.values:
    if df['neighborhood'][i] == 'missing':
        df.loc[i, 'neighborhood'] = df['zip_code'][i]

#condense some under-represented neighborhoods
value_counts = df['neighborhood'].value_counts() # Specific column 
to_replace = value_counts[value_counts < 10].index
df['neighborhood'].replace(to_replace, '', inplace=True)

#delete zip code since we already have neighborhood and elementary school
del df['zip_code']



# convert each sale date to month key
df = df[df['sale_date'] != '2017-07-01'] #get rid of last days' sales so we have full months
df['sale_date'] = pd.to_datetime(df['sale_date'])
df['sale_month'] = df['sale_date'].dt.month
df['sale_month'].astype(int)
del df['sale_date']

#we need to create a sale month key so that months that are near each other 
#are close in numeric value (since i have 7/16 - 6/17 instead of 1/16-12/16)
df['sale_month'].replace(7, 20, inplace = True)
df['sale_month'].replace(8, 21, inplace = True)
df['sale_month'].replace(9, 22, inplace = True)
df['sale_month'].replace(10, 23, inplace = True)
df['sale_month'].replace(11, 24, inplace = True)
df['sale_month'].replace(12, 25, inplace = True)
df['sale_month'].replace(1, 26, inplace = True)
df['sale_month'].replace(2, 27, inplace = True)
df['sale_month'].replace(3, 28, inplace = True)
df['sale_month'].replace(4, 29, inplace = True)
df['sale_month'].replace(5, 30, inplace = True)
df['sale_month'].replace(6, 31, inplace = True)
df = df.rename(index=int, columns={"sale_month": "sale_month_key"})


#convert bathrooms to numeric values
df['plumbing'].replace('TWO FULL BATHS', 2, inplace = True)
df['plumbing'].replace('TWO FULL BATHS, TWO HALF BATHS', 3, inplace = True)
df['plumbing'].replace('FIVE FULL BATHS, TWO HALF BATHS', 6, inplace = True)
df['plumbing'].replace('SIX FULL BATHS, THREE HALF BATHS', 7.5, inplace = True)
df['plumbing'].replace('THREE FULL BATHS, FOUR HALF BATHS', 5, inplace = True)
df['plumbing'].replace('ONE HALF BATH', 0.5, inplace = True)
df['plumbing'].replace('EIGHT FULL BATHS', 0.5, inplace = True)
df['plumbing'].replace('ONE FULL BATH, ONE HALF BATH', 1.5, inplace = True)
df['plumbing'].replace('ONE FULL BATH', 1, inplace = True)
df['plumbing'].replace('THREE FULL BATHS, ONE HALF BATH', 3.5, inplace = True)
df['plumbing'].replace('TEN FULL BATHS, HB', 10, inplace = True)
df['plumbing'].replace('TWO HALF BATHS', 1, inplace = True)
df['plumbing'].replace('THREE FULL BATHS', 3, inplace = True)
df['plumbing'].replace('ONE HALF BATH, ONE FULL BATH', 1.5, inplace = True)
df['plumbing'].replace('FOUR FULL BATHS, FOUR HALF BATHS', 6, inplace = True)
df['plumbing'].replace('TWO FULL BATHS, FIVE HALF BATHS', 4.5, inplace = True)
df['plumbing'].replace('ONE FULL BATH, TWO HALF BATHS', 2, inplace = True)
df['plumbing'].replace('FIVE FULL BATHS, THREE HALF BATHS', 6.5, inplace = True)
df['plumbing'].replace('THREE FULL BATHS, TWO HALF BATHS', 4, inplace = True)
df['plumbing'].replace('FOUR FULL BATHS', 4, inplace = True)
df['plumbing'].replace('FIVE FULL BATHS', 5, inplace = True)
df['plumbing'].replace('TWO FULL BATHS, ONE HALF BATH', 2.5, inplace = True)
df['plumbing'].replace('TWO FULL BATHS, THREE HALF BATHS', 3.5, inplace = True)
df['plumbing'].replace('SIX FULL BATHS, ONE HALF BATH', 6.5, inplace = True)
df['plumbing'].replace('ONE HALF BATH, THREE FULL BATHS', 3.5, inplace = True)
df['plumbing'].replace('FIVE FULL BATHS, ONE HALF BATH', 5.5, inplace = True)
df['plumbing'].replace('ONE HALF BATH, TWO FULL BATHS', 2.5, inplace = True)
df['plumbing'].replace('FOUR FULL BATHS, TWO HALF BATHS', 5, inplace = True)
df['plumbing'].replace('FOUR FULL BATHS, ONE HALF BATH', 4.5, inplace = True)
df['plumbing'].replace('THREE FULL BATHS, THREE HALF BATHS', 4.5, inplace = True)
df['plumbing'].replace('ONE FULL BATH, THREE HALF BATHS', 2.5, inplace = True)
df['plumbing'].replace('SIX FULL BATHS, TWO HALF BATHS', 7, inplace = True)

df['plumbing'] = df['plumbing'].astype(float)



#condense options for various zillow descriptions
for i in df.index.values:
    if 'Radiant' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Radiant'
    if 'Geothermal' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Geothermal'
    if 'Heat pump' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Heat Pump'
    if 'Forced air' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Forced air'
    if 'Baseboard' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Baseboard'
    if 'Wall' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Wall'
    if 'Stove' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Stove'
    if 'Other' in df['heatingsystem'][i]:
        df.loc[i, 'heatingsystem'] = 'Other'
        
        
for i in df.index.values:
    if 'Central' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Central'
    if 'Solar' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Solar'
    if 'Evaporation' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Evaporation'
    if 'Geothermal' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Geothermal'
    if 'Refrigeration' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Refrigeration'
    if 'Wall' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Wall'
    if 'Other' in df['coolingsystem'][i]:
        df.loc[i, 'coolingsystem'] = 'Other'
        
for i in df.index.values:
    if 'Tile' in df['roof'][i]:
        df.loc[i, 'roof'] = 'Tile'
    if 'Built-up' in df['roof'][i]:
        df.loc[i, 'roof'] = 'Built-up'
    if 'Composition' in df['roof'][i]:
        df.loc[i, 'roof'] = 'Composition'
    if 'Shake / Shingle' in df['roof'][i]:
        df.loc[i, 'roof'] = 'Shake / Shingle'
    if 'Asphalt' in df['roof'][i]:
        df.loc[i, 'roof'] = 'Asphalt'
        
for i in df.index.values:
    if 'Vinyl' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Vinyl'
    if 'Cement / Concrete' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Cement / Concrete'    
    if 'Stucco' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Stucco'  
    if 'Brick' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Brick'  
    if 'Stone' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Stone'  
    if 'Metal' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Metal'  
    if 'Shingle' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Shingle'
    if 'Wood products' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Wood products'  
    if 'Composition' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Composition' 
    if 'Wood' in df['exteriormaterial'][i]:
        df.loc[i, 'exteriormaterial'] = 'Wood' 
        
for i in df.index.values:
    if 'Carport' in df['parkingtype'][i]:
        df.loc[i, 'parkingtype'] = 'Carport'
    if 'Garage - Attached' in df['parkingtype'][i]:
        df.loc[i, 'parkingtype'] = 'Garage - Attached'
    if 'Garage - Detached' in df['parkingtype'][i]:
        df.loc[i, 'parkingtype'] = 'Garage - Detached'
    if 'Off-street' in df['parkingtype'][i]:
        df.loc[i, 'parkingtype'] = 'Off-street'


#clean up elementary school typos 
df['elementaryschool'] = df['elementaryschool'].str.lower()
df['elementaryschool'] = df['elementaryschool'].str.strip()

for i in df.index.values:
    if 'school' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('school', '')
    if 'elementary' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('elementary', '')
    if 'k-8' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('k-8', '')
    if 'environmental' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('environmental', 'env')
    if 'elementry' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('elementry', 'elementary')
    if 'campus' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('campus', '')
    if 'hollyrood' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('hollyrood', '')
    if 'laurelhust' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('laurelhust', 'laurelhurst')
    if 'abernathy' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('abernathy', 'abernethy')
    if 'beverly clearly-' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('beverly clearly-', 'beverly cleary')
    if 'beverly cleary-' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('beverly cleary-', 'beverly cleary')
    if 'laurelhurst ()' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('laurelhurst ()', 'laurelhurst')
    if 'laurelhurst k-6 moving to' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('laurelhurst k-6 moving to', 'laurelhurst')
    if 'laurelhurst, all saints' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('laurelhurst, all saints', 'laurelhurst') 
    if 'elem' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('elem', '')
    if 'buckman arts magnet' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('buckman arts magnet', 'buckman')
    if 'abernethy/winterhaven' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('abernethy/winterhaven', 'abernethy')
    if 'boise/eliot' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('boise/eliot', 'boise-eliot')
    if 'bridelmile' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('bridelmile', 'bridlemile')
    if 'bridelmile' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('bridelmile', 'bridlemile')
    if 'bridemile' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('bridemile', 'bridlemile')
    if 'humbolt' in df['elementaryschool'][i]:
        df.loc[i, 'elementaryschool'] = df.loc[i, 'elementaryschool'].replace('humbolt', 'humboldt')

#condense some under-represented elementary schools
value_counts = df['elementaryschool'].value_counts() # Specific column 
to_replace = value_counts[value_counts < 11].index
df['elementaryschool'].replace(to_replace, '', inplace=True)



#save this data set
df.to_pickle('single_family_homes.pkl')
df.to_csv('single_family_homes.csv')
df.shape


#create a one-hot matrix for categorical variables
columns = ['type', 'neighborhood',  'roof', 'parkingtype', 'exteriormaterial', 'heatingsystem', 'coolingsystem', 'architecture', 'elementaryschool']
for column in columns:
    one_hot_matrix = patsy.dmatrix(column,data=df,return_type='dataframe')
    df.drop([column], axis=1, inplace=True)
    df = df.join(one_hot_matrix)
    del df['Intercept']
    
#just remember that you need to delete addresses, sale price, and homedescription before you perform regression


#save the one-hot data set
df.to_pickle('single_family_homes_one_hot.pkl')
df.shape


