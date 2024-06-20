import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
import os


def build_url():
    
    place = input("Please enter the name of the place (city, State) you want to search restaurants in (e.g. \"Fremont, CA\"): ")
    lst = [x.strip() for x in place.split(',')]
    if len(lst[0].split())>1:
        lst[0] ='+'.join(lst[0].split())
    
    baseurl = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc='
    url = baseurl +lst[0]+',+'+lst[1]
    
    return url


def query_restaurant(num_restaurant=10):
    
    import urllib.request, urllib.parse, urllib.error
    from bs4 import BeautifulSoup
    import ssl
    import pandas as pd
    
    num_loop_restaurant = 1+int(num_restaurant/11)
    
    url = build_url()
    
    if num_loop_restaurant==1:
        soup=read_soup_HTML(url)
        restaurant_names = build_restaurant_names(soup)
        restaurant_links = build_restaurant_links(soup)
    else:
        soup=read_soup_HTML(url)
        restaurant_names = build_restaurant_names(soup)
        restaurant_links = build_restaurant_links(soup)
        for i in range(1,num_loop_restaurant):
            url = url+'&start='+str(i*10)
            soup=read_soup_HTML(url)
            restaurant_names.extend(build_restaurant_names(soup))
            restaurant_links.extend(build_restaurant_links(soup))
    
    df=pd.DataFrame(data={'Link':restaurant_links,'Name':restaurant_names})
    print(df.iloc[:num_restaurant])
    
    return df.iloc[:num_restaurant]


def read_soup_HTML(url):
    
    import urllib.request, urllib.parse, urllib.error
    from bs4 import BeautifulSoup
    import ssl

    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Read the HTML from the URL and pass on to BeautifulSoup
    #print("Opening the page", url)
    uh= urllib.request.urlopen(url, context=ctx)
    html =uh.read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def build_restaurant_names (soup):
    restaurant_names = []
    for span in soup.find_all('span'):
        if 'class' in span.attrs:
            if span.attrs['class']==['indexed-biz-name']:
                restaurant_names.append(span.contents[1].get_text())
    
    return restaurant_names


def build_restaurant_links (soup):
    restaurant_links=[]
    for a in soup.find_all('a'):
        if 'class' in a.attrs:
            #print(a.attrs)
            if a.attrs['class']==['js-analytics-click']:
                restaurant_links.append(a.attrs['href'])
    _=restaurant_links.pop(0)
    
    for i in range(len(restaurant_links)):
        link='https://yelp.com'+restaurant_links[i]
        restaurant_links[i]=link
    
    return restaurant_links


query_restaurant(num_restaurant=15)


def gather_reviews(df,num_reviews):
    
    reviews={}
    num_links=df.shape[0]
    num_loop_reviews = 1+int(num_reviews/21)
    for i in range(num_links):
        print(f"Gathering top reviews on {df.iloc[i]['Name']} now...")
        if num_loop_reviews==1:
            review_text=[]
            url=df.iloc[i]['Link']
            soup=read_soup_HTML(url)
            for p in soup.find_all('p'):
                if 'itemprop' in p.attrs:
                    if p.attrs['itemprop']=='description':
                        text=p.get_text().strip()
                        review_text.append(text)
        else:
            review_text=[]
            url=df.iloc[i]['Link']
            soup=read_soup_HTML(url)
            for p in soup.find_all('p'):
                if 'itemprop' in p.attrs:
                    if p.attrs['itemprop']=='description':
                        text=p.get_text().strip()
                        review_text.append(text)
            for i in range(1,num_loop_reviews):
                url=df.iloc[i]['Link']+'?start='+str(20*i)
                soup=read_soup_HTML(url)
                for p in soup.find_all('p'):
                    if 'itemprop' in p.attrs:
                        if p.attrs['itemprop']=='description':
                            text=p.get_text().strip()
                            review_text.append(text)
        
        reviews[str(df.iloc[i]['Name'])]=review_text[:num_reviews]
    
    return reviews


def get_reviews(num_restaurant=10,num_reviews=20):
    df_restaurants = query_restaurant(num_restaurant=num_restaurant)
    reviews = gather_reviews(df_restaurants,num_reviews=num_reviews)
    
    return reviews


# ### Test cases
# 

rev = get_reviews(5,5)


count=0
for r in rev['The Table']:
    print(r)
    print("="*100)
    count+=1
print("\n Review count:", count)





# ### Import SQLite library of Python (it is built-in)
# * Create a connection
# * Create a cursor object
# 

import sqlite3
conn = sqlite3.connect('emaildb.sqlite')
cur=conn.cursor()


# ### Run SQL commands through the cursor
# * Drop the previous Table if it exists
# * Create a Table with email and counts as attributes
# 

cur.execute('DROP TABLE IF EXISTS Counts')
cur.execute('''
CREATE TABLE Counts (email TEXT, count INTEGER)''')


# ### Use urllib method to establish a connection with a remote server to read the inbox data
# 

import urllib.request, urllib.parse, urllib.error

print("Opening the file connection...")
# Following example reads Project Gutenberg EBook of Pride and Prejudice
fhand = urllib.request.urlopen('http://data.pr4e.org/mbox.txt')


# ### Read the text data from the remote server to create a local text file
# 

txt_dump = ''
line_count=0
word_count=0
# Iterate over the lines in the file handler object and dump the data into the text string. 
# Also increment line and word counts
for line in fhand:
# Use decode method to convert the UTF-8 to Unicode string
    txt_dump+=line.decode()
    line_count+=1
    # Count the length of words in the line and add to the running count
    word_count+=len(line.decode().split(' '))


# Prints basic informationn about the text data
print("Printing some info on the text dump\n"+"-"*60)
print("Total characters:",len(txt_dump))
print("Total words:",word_count)
print(f"Total lines: {line_count}")


# ### Open a local file handler with the text file
# 

file = open('mbox.txt','w') 
file.write(txt_dump)
file.close()


fh=open('mbox.txt')


# ### Show first few lines of the text data
# 

show_text=fh.read(1000)
print(show_text)


# ### Read the text file line by line to extract the email address and INSERT INTO/UPDATE the SQL Table 
# 

for line in fh:
    if not line.startswith('From: '): continue
    pieces = line.split()
    email = pieces[1]
    cur.execute('SELECT count FROM Counts WHERE email = ? ', (email,))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (email, count)
                VALUES (?, 1)''', (email,))
    else:
        cur.execute('UPDATE Counts SET count = count + 1 WHERE email = ?',
                    (email,))
conn.commit()


# ### Execute SQL command to read email count from the database and ORDER BY the count
# 

sqlstr = 'SELECT email,count FROM Counts ORDER BY count DESC LIMIT 20'

for row in cur.execute(sqlstr):
    print(str(row[0]), row[1])


# ### Run AVG command with a LIKE matching string to count the average number of emails from a particular source
# 

sqlstr = 'SELECT AVG(count) FROM Counts WHERE email LIKE "%umich%"'
for row in cur.execute(sqlstr):
    print(float(row[0]))


# #### Close the remote and local file handler and the cursor connection
# 

fh.close()
cur.close()
fhand.close()


# ## This notebook retrieves the weather data of a town/city (input by user)
# 

import urllib.request, urllib.parse, urllib.error
import json


# ### Gets the secret API key (you have to get one from Openweather website and use) from a JSON file, stored in the same folder
# 

with open('APIkeys.json') as f:
    keys = json.load(f)
    weatherapi = keys['weatherapi']


serviceurl = 'http://api.openweathermap.org/data/2.5/weather?'
apikey = 'APPID='+weatherapi


# ### Runs a loop of getting address input from user and prints basic weather data
# 

while True:
    address = input('Enter the name of a town (enter \'quit\' or hit ENTER to quit): ')
    if len(address) < 1 or address=='quit': break

    url = serviceurl + urllib.parse.urlencode({'q': address})+'&'+apikey
    print(f'Retrieving the weather data of {address} now... ')
    uh = urllib.request.urlopen(url)
    
    data = uh.read()
    json_data=json.loads(data)
    
    main=json_data['main']
    description = json_data['weather'][-1]['description']
    
    pressure_mbar = main['pressure']
    pressure_inch_Hg = pressure_mbar*0.02953
    humidity = main['humidity']
    temp_min = main['temp_min']-273
    temp_max = main['temp_max']-273
    temp = main['temp']-273
    
    print(f"\nRight now {address} has {description}. Key weather parameters are as follows\n"+"-"*70)
    print(f"Pressure: {pressure} mbar/{pressure_inch_Hg} inch Hg")
    print(f"Humidity: {humidity}%")
    print(f"Temperature: {round(temp,2)} degree C")





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
import os


def build_url_place(place=None):
    
    if place==None:
        place = input("Please enter the name of the place (city, State) you want to search restaurants in (e.g. \"Fremont, CA\"): ")
    
    if ',' in place:
        lst = [x.strip() for x in place.split(',')]
        if len(lst[0].split())>1:
            lst[0] ='+'.join(lst[0].split())

        baseurl = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc='
        url = baseurl +lst[0]+',+'+lst[1]
    else:
        if len(place.split())>1:
            place ='+'.join(place.split())

        baseurl = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc='
        url = baseurl +place
    
    return (url,place)


def build_url_zip(zipcode=None):
    
    if zipcode==None:
        zipcode = input("Please enter the 5 digit zipcode (US) you want to search restaurants around: ")
    
    baseurl = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc='
    url = baseurl +str(zipcode)
    
    return (url,zipcode)


def query_restaurant_place(num_restaurant,place=None,verbosity=1):
    
    import urllib.request, urllib.parse, urllib.error
    from bs4 import BeautifulSoup
    import ssl
    import pandas as pd
    
    num_loop_restaurant = 1+int(num_restaurant/11)
    
    if place==None:
        url,_ = build_url_place()
    else:
        url,_ = build_url_place(place)
    
    if num_loop_restaurant==1:
        soup=read_soup_HTML(url)
        restaurant_names = build_restaurant_names(soup)
        restaurant_links = build_restaurant_links(soup)
    else:
        soup=read_soup_HTML(url)
        restaurant_names = build_restaurant_names(soup)
        restaurant_links = build_restaurant_links(soup)
        for i in range(1,num_loop_restaurant):
            url = url+'&start='+str(i*10)
            soup=read_soup_HTML(url)
            restaurant_names.extend(build_restaurant_names(soup))
            restaurant_links.extend(build_restaurant_links(soup))
    
    df=pd.DataFrame(data={'Link':restaurant_links,'Name':restaurant_names})
    if verbosity==1:
        print("\n Top restaurants found\n"+"-"*100)
        for name in restaurant_names[:num_restaurant]:
            print(name,end=', ')
    
    return df.iloc[:num_restaurant]


def query_restaurant_zip(num_restaurant,zipcode=None,verbosity=1):
    
    import urllib.request, urllib.parse, urllib.error
    from bs4 import BeautifulSoup
    import ssl
    import pandas as pd
    
    num_loop_restaurant = 1+int(num_restaurant/11)
    
    if zipcode==None:
        url,_ = build_url_zipcode()
    else:
        url,_ = build_url_zip(zipcode)
    
    if num_loop_restaurant==1:
        soup=read_soup_HTML(url)
        restaurant_names = build_restaurant_names(soup)
        restaurant_links = build_restaurant_links(soup)
    else:
        soup=read_soup_HTML(url)
        restaurant_names = build_restaurant_names(soup)
        restaurant_links = build_restaurant_links(soup)
        for i in range(1,num_loop_restaurant):
            url = url+'&start='+str(i*10)
            soup=read_soup_HTML(url)
            restaurant_names.extend(build_restaurant_names(soup))
            restaurant_links.extend(build_restaurant_links(soup))
    
    df=pd.DataFrame(data={'Link':restaurant_links,'Name':restaurant_names})
    
    if verbosity==1:
        print("\n Top restaurants found\n"+"-"*100)
        for name in restaurant_names[:num_restaurant]:
            print(name,end=', ')
    
    return df.iloc[:num_restaurant]


def read_soup_HTML(url):
    
    import urllib.request, urllib.parse, urllib.error
    from bs4 import BeautifulSoup
    import ssl

    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Read the HTML from the URL and pass on to BeautifulSoup
    #print("Opening the page", url)
    uh= urllib.request.urlopen(url, context=ctx)
    html =uh.read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def build_restaurant_names (soup):
    restaurant_names = []
    for span in soup.find_all('span'):
        if 'class' in span.attrs:
            if span.attrs['class']==['indexed-biz-name']:
                restaurant_names.append(span.contents[1].get_text())
    
    return restaurant_names


def build_restaurant_links (soup):
    restaurant_links=[]
    for a in soup.find_all('a'):
        if 'class' in a.attrs:
            #print(a.attrs)
            if a.attrs['class']==['js-analytics-click']:
                restaurant_links.append(a.attrs['href'])
    _=restaurant_links.pop(0)
    
    for i in range(len(restaurant_links)):
        link='https://yelp.com'+restaurant_links[i]
        restaurant_links[i]=link
    
    return restaurant_links


def gather_reviews(df,num_reviews,verbosity=1):
    
    reviews={}
    num_links=df.shape[0]
    num_loop_reviews = 1+int(num_reviews/21)
    
    if verbosity==1:
        print("\n")
    
    for i in range(num_links):
        if verbosity==1:
            print(f"Gathering top reviews on {df.iloc[i]['Name']} now...")
        
        if num_loop_reviews==1:
            review_text=[]
            url=df.iloc[i]['Link']
            soup=read_soup_HTML(url)
            for p in soup.find_all('p'):
                if 'itemprop' in p.attrs:
                    if p.attrs['itemprop']=='description':
                        text=p.get_text().strip()
                        review_text.append(text)
        else:
            review_text=[]
            url=df.iloc[i]['Link']
            soup=read_soup_HTML(url)
            for p in soup.find_all('p'):
                if 'itemprop' in p.attrs:
                    if p.attrs['itemprop']=='description':
                        text=p.get_text().strip()
                        review_text.append(text)
            for i in range(1,num_loop_reviews):
                url=df.iloc[i]['Link']+'?start='+str(20*i)
                soup=read_soup_HTML(url)
                for p in soup.find_all('p'):
                    if 'itemprop' in p.attrs:
                        if p.attrs['itemprop']=='description':
                            text=p.get_text().strip()
                            review_text.append(text)
        
        reviews[df.iloc[i]['Name']]=review_text[:num_reviews]
        if verbosity==1:
            print(f"Reviews for {df.iloc[i]['Name']} gathered.\n"+"-"*60)
    
    return reviews


def get_reviews_place(num_restaurant=10,num_reviews=20,place=None,verbosity=1):
    
    if place==None:
        df_restaurants = query_restaurant_place(num_restaurant=num_restaurant,verbosity=verbosity)
    else:
        df_restaurants = query_restaurant_place(num_restaurant=num_restaurant,place=place,verbosity=verbosity)
    
    reviews = gather_reviews(df_restaurants,num_reviews=num_reviews,verbosity=verbosity)
    
    return reviews


def get_reviews_zip(num_restaurant=10,num_reviews=20,zipcode=None,verbosity=1):
    
    if zipcode==None:
        df_restaurants = query_restaurant_zip(num_restaurant=num_restaurant)
    else:
        df_restaurants = query_restaurant_zip(num_restaurant=num_restaurant,zipcode=zipcode)
    
    reviews = gather_reviews(df_restaurants,num_reviews=num_reviews,verbosity=verbosity)
    
    return reviews


# ### Test cases
# 

rev = get_reviews_place(num_restaurant=5,num_reviews=15,place='Chicago, IL',verbosity=0)


rev.keys()


rev = get_reviews_zip(num_restaurant=5,num_reviews=15,zipcode=95129)


# ## Word Cloud generation
# 

def wordcloud_from_text(text):
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    
    stopwords = set(STOPWORDS)
    more_stopwords=['food','good','bad','came','place','restaurant','really','much','less','more']
    for word in more_stopwords:
        stopwords.add(word)

    wc = WordCloud(background_color="white", max_words=50, stopwords=stopwords,max_font_size=40)
    _=wc.generate(text)
    
    plt.figure(figsize=(10,7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def wordcloud_from_reviews(review_dict):
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    
    stopwords = set(STOPWORDS)
    more_stopwords=['food','good','bad','came','place','restaurant','really','much','less','more']
    for word in more_stopwords:
        stopwords.add(word)

    wc = WordCloud(background_color="white", max_words=50, stopwords=stopwords,max_font_size=40)
    
    for restaurant in review_dict:
        text = '\n'.join(review_dict[restaurant])
        _= wc.generate(text)
        
        plt.figure(figsize=(10,7))
        plt.title(f"Wordcloud for {restaurant}\n",fontsize=20)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()


def plot_wc(wc,place=None, restaurant=None):
    plt.figure(figsize=(12,8))
    
    if place!=None:
        plt.title("{}\n".format(place),fontsize=20)
        
    if restaurant!=None:
        plt.title("{}\n".format(restaurant),fontsize=20)
    
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def wordcloud_city(place=None,num_restaurant=10,num_reviews=20,stopword_list=None,
                   disable_default_stopwords=False,verbosity=1):
    
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    
    if place==None:
        review_dict=get_reviews_place(num_restaurant=num_restaurant,num_reviews=num_reviews,verbosity=verbosity)
    else:
        review_dict=get_reviews_place(num_restaurant=num_restaurant,num_reviews=num_reviews,
                                      place=place,verbosity=verbosity)
    
    text=""
    
    for restaurant in review_dict:
        text_restaurant = '\n'.join(review_dict[restaurant])
        text+=text_restaurant
    
    # Add custom stopwords to the default list
    stopwords = set(STOPWORDS)
    more_stopwords=['food','good','bad','best','amazing','go','went','came','come','back','place','restaurant',
                    'really','much','less','more','order','ordered','great','time','wait','table','everything',
                   'take','definitely','sure','recommend','recommended','delicious','taste','tasty',
                   'menu','service','meal','experience','got','night','one','will','made','make',
                    'bit','dish','dishes','well','try','always','never','little','big','small', 'nice','excellent']
    if not disable_default_stopwords:
        for word in more_stopwords:
            stopwords.add(word)
    if stopword_list!=None:
        for word in stopword_list:
            stopwords.add(word)

    wc = WordCloud(background_color="white", max_words=50, stopwords=stopwords,max_font_size=40,scale=3)
    _= wc.generate(text)
    
    plot_wc(wc,place=place)
    
    return wc


# ### Testing world cloud
# 

wordcloud_from_reviews(rev)


wordcloud_city(place='Palo Alto, CA',num_restaurant=25)


wordcloud_city(place='Chicago',num_restaurant=25)


wc=wordcloud_city(place='San Jose, California',num_restaurant=5,num_reviews=5,verbosity=0)


# ## Top 20 US cities food scene word cloud
# 

cities=pd.read_html("http://www.citymayors.com/gratis/uscities_100.html")


cities_list=list(cities[1][1])


cities_list.pop(0)


Top_cities=[x.split(';')[0] for x in cities_list[:20]]


Top_cities


for city in Top_cities[:5]:
    wordcloud_city(city,verbosity=0,num_restaurant=100)





