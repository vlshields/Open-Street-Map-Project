# Open Street Maps Case Study
********************************************************************************************************************

### Map Location

>San Francisco, CA, USA

> * http://www.openstreetmap.org/search?query=San%20Francisco%2C%20CA#map=12/37.7746/-122.4193
> * https://mapzen.com/data/metro-extracts/metro/san-francisco_california/

>I grew up about 30 miles south of San Francisco, and I've always been interested in the city. I'm very interested to see if various sqlite database queries match my expectations. Additionally, I would like to help with the improvement of San Francisco Open Street Map data in any way I can.

# Problems encountered
********************************************************************************************************************

The most challenging problem encountered during the initial audit was the sheer size of the file. The orginal san_francisco_osm file turned out to be 1.41 GB uncompressed. When ran against data.py, the program did not finish running, even when left alone overnight. For this reason, san_francisco_osm was split into a sample xml file (about 80MB). The file was split using the code provided in the 'Project Details' page for this project. The sample osm file was then parsed into its respective csv files and validated with the cerberus module. Once the sample file was validated without error, the original osm file was parsed and the validation was set to False. This allowed the program to run much more quickly. In addition to the aformentioned challenge, a few other errors and inconsistencies were found during the preliminary audit. These problems are listed below and will be detailed in this section.

* Special characters in the street name that don't seem to belong ('16th St #404')


* Values that pass the 'is_street_name' test but seem to be an address rather than street ('N Side Of Foothill Blvd @ John Dr (Near I-580)'


* Extra whitespace (markdown won't render the whitespace, so I left the examples out)


* Inconsistencies in formatting of k tags taken from tiger gps data

Additionally, this audit revealed some values who's formatting were not necessarily erroneous, but were inconsistent with the rest of the data. For example, in the case of street names, the suffix 'St' appears many times while the suffix 'Street' appears also. The same goes for street names like 'Blvd' and 'Rd'. For the sake of consistency within the SQL database, these issues were cleaned with a custom function:
```python
def update_name(name, mapping):
    m = street_type_re.search(name)  # Extract street type from street name
    if m:
        if m.group() in mapping:  # Check if the street type is in the mapping dictionary
            return name.replace(m.group(), mapping[m.group()])
    return name
```
Here, string replacement was used rather than the re.sub method. This function was called inside the shape_element function (from data.py) so that all cleaning was taken care of as the data was formatted to their respective csv files.

### Special Characters

The results of the initial audit revealed several instanses of special characters, especially in street names. Although the special characters (#,@) discovered do not seem to be random, they do not belong in a street name. This is especially true for the pound character, because usually the character denotes a unit in a building. Perhaps these values are misplaced? Regular expressions and the re module were used to drop any value containing these characters in the shape_element function from data.py in the following manner:

```python
if not PROBLEMCHARS.search(tags_dict['key']):
                tags.append(tags_dict)
```

A review of the results that followed revealed that this also took care of the strange address/street name inconsistencies, especially since many of the problematic street names contained a  '@' character. These values were also dropped to create more consistency within the data.

### Extra whitespace

The pattern in the variable PROBLEMCHARS was compiled in the following manor:

```python
import re
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

```


A look at the re documentation reveals that this pattern will also match with any value where there is extra whitespace. 

### Tiger GPS inconsistencies

To handle the formatting inconcistency between data taken from Tiger GPS and the rest of the data, an extra field called 'type' was created in the tags(nodes_tags, ways_tags) csvs. Each value was checked for a ':' character, and if there was one, the value was split on the colon character. The value after the colon character was set to the 'type' field (this, presumably would be 'tiger') the value before the colon was set to the 'key' field. If no colon was present, the 'type' field was entered as 'regular'. This also proved useful for other values that contain colon characters, such as "addr:street".
```python
if ':' in tag.attrib['k']:
                tags_dict['key'] = tag.attrib['k'].split(':', 1)[1:][0]  
                tags_dict['type'] = tag.attrib['k'].split(':', 1)[:1][0] 
            else:                                                   
                tags_dict['key'] = tag.attrib['k']                 
                tags_dict['type'] = 'regular'
```
 
# Data Overview 
*********************************************************************************************************************

This section details various statistics on the San Francisco open street map data as well as the sqlite queries used to determine said statistics. It also provides some additional ideas for further study.

#### File sizes

| Files        | Size           | 
| ------------- |:-------------| 
| san-franciso_california.osm      | 1.41 GB | 
| nodes_tags.csv    | 438 KB     |  
| nodes.csv | 555.2 MB  | 
| san_francisco_osm.db | 969.1 MB|
| ways_nodes.csv | 189.2 MB |
| ways_tags.csv | 45.7 MB |
| ways.csv | 50.4 MB |


#### Number of nodes
```*.sql
SELECT count(*) FROM nodes;
6617994
```


#### Number of ways
```*.sql
SELECT count(*) FROM ways;
823438
```

#### Top ten contributing users

```*.sql
SELECT nt.user, COUNT(*) as num
FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) nt
GROUP BY nt.user
ORDER BY num DESC
LIMIT 10;

andygol,1496265
ediyes,887682
Luis36995,679938
dannykath,545936
RichRico,415823
Rub21,383497
calfarome,190923
oldtopos,165667
KindredCoda,148172
karitotp,139745
```
#### Number of unique users

```*.sql
SELECT COUNT(DISTINCT(user)) 
FROM 
( SELECT user FROM nodes UNION ALL SELECT user FROM ways) nt;

2787
```
#### Number of users having five or less posts

```*.sql
SELECT COUNT(*) FROM 
(SELECT nt.user, COUNT(*) as num FROM 
(SELECT user FROM nodes UNION ALL SELECT user FROM ways) nt 
GROUP BY nt.user HAVING num < 5) nt;
1175
```
# Summarizing thoughts
*********************************************************************************

I was actually quite suprised by the results from the various user queries ran on the San Francisco OSM database. Frankly, I expected user names in the top ten contributing users to appear much more 'bot-like'. I have examined other osm files, and I have seen many instances of 'woodpeck_fixbot' or 'TIGERcn1'. My confusion and curiosity about this issue led me to run a query like this:

```*.sql
SELECT nt.user FROM 
(SELECT user FROM 
nodes UNION ALL SELECT user FROM 
ways) nt WHERE nt.user LIKE '%bot%';
```
And 'woodpeck_fixbot' was indeed a value that appears in the user field (along with 'xybot', 'dolceBottle', 'Rococoroboto',  and 'dacheatbot') for one or both of the tables queried. That it did not make it in the top ten contributing user list vexed me even more, so I continued to explore this issue:

```*.sql
SELECT nt.user, COUNT(*) FROM 
(SELECT user FROM 
nodes UNION ALL SELECT user FROM 
ways) nt WHERE user = 'woodpeck_fixbot';

woodpeck_fixbot|24977
```
Now it makes sense why 'woodpeck_fixbot' did not make it in the top ten list. For the open street maps project, I first looked at the city of Carmel, CA, but the file turned out to be too small. In that dataset, 'woodpeck_fixbot' was by far the top contributing user, so I was again surprised by the results of the query above. This leads me to believe that the users in the top ten list are in fact bots. There is no way to *prove* this, of course, but the evidence seems strong. How could a human user outpreform a bot that was programmed to update osm data. It would have to be their whole job if this was the case. That being said, some of the user names in the top ten list appear very human like, 'Luis36995' for example or 'dannykath'. The latter of the two seems like it could be a first and last name. Looking at a user's username, however, is not a very comprehensive way to determine wether or not they are a human being.

#### Some user statistics


* The top contributor 'andygol' makes up about 20.1% of user contributions


* The top ten contributors make up 68% of the user contributions


* The number of users having five or less posts makes up 0.01% of user contributions.


* Average number of user contributions: 2670.0509508432 posts.


Another aspect of user contribution that surprises me is the spread. I expected to see the top ten contributing users to make up at least 80 percent of the overall posts, but they turned out to be a little less than 70 percent. In addition, I expected the number of users having five or less posts to make up at least 1 percent of user contributions. This implies that about 47.98 percent of contributers are somewhere in the middle, cooberating the theory that a high percentage of contributing users are bots. If more users were humans and not bots, I would expect to see the top ten contributing users to make up a larger percentage of total contributions, and a smaller average number of user contributions. However, drawing conclusions here would be premature; there is no real way to know.  

# Some Further Exploration
************************************************************************************************************

#### Top ten amenities

```*.sql
SELECT value, COUNT(*) as count 
FROM (SELECT * FROM nodes_tags UNION ALL SELECT * FROM ways_tags) nt 
WHERE key = 'amenity' GROUP BY value 
ORDER BY count DESC LIMIT 10;

parking,4597
restaurant,3404
school,1294
bench,1217
place_of_worship,1173
cafe,1080
fast_food,715
post_box,688
bicycle_parking,584
drinking_water,532
```
Anyone who has ever been to San Francisco should not be surprised that parking is the number 1 amenity, by about 6 percentage points. Restaurants make up about 16% of total amenities while schools make up about 6% of the total amenities listed. Is this cause for sadness? I think the ratio could be worse. Its good to see that schools are at least the third most frequent amenity, even though it only beats out 'place_of_worship' by a little less than two thirds of a percentage point. Also, if you've been to San Francisco you should not be surprised about the frequency of bicycle parking. I am reminded of the bay area bike share analysis that we saw at the beginning of the DAND course. There is a new start up in SF called "Scoot", in which people rent motorized scooters and return them after their workday. I wonder if the scooter parking is considered bicycle parking in the Open Street Map data.

#### Bank popularity

```*.sql
SELECT nodes_tags.value, COUNT(*) as num 
FROM nodes_tags JOIN (SELECT DISTINCT(id) 
FROM nodes_tags WHERE value='bank') i ON 
nodes_tags.id=i.id WHERE nodes_tags.key='name' 
GROUP BY nodes_tags.value ORDER BY num DESC;

Chase,42
Citibank,26
HSBC,5
citibank,3
CitiBank,2
USBank,2
UnionBank,2
Citybank,1
Comerica,1
Wachovia,1
usbank,1
```
This was mostly just for curiosities' sake, and I thought I'd share the results in this report. I was curious to do a query relating to bank data because San Francisco has a reputation of being a financial city. However, I suppose there must be some mistake, because I would expect to see wells fargo somewhere in this list. Perhaps this is evidence that the Open Street map data for San Francisco is incomplete? As you can see here also, Citibank appears in several different text formats, so in reality this table should read 'Citibank, 32'. I was tempted to name this section "the curiuos case of the missing wells fargos" but I'll save the comedy. I suppose its not really a joking matter, because something seems wrong here. Anyway, if frequency is a good heuristic for bank popularity, then Chase seems relatively popular. Banks are one area of the San Francisco Open Street Map data that could really be improved for uniformity.

#### Is there nature in San Francisco?
```*.sql
SELECT nt.value, count(*) as num FROM 
(SELECT * FROM nodes_tags UNION ALL SELECT 
* FROM ways_tags) nt WHERE key = 'natural' 
GROUP BY nt.value ORDER BY num DESC LIMIT 10;

tree,3824
water,416
wood,298
sand,254
coastline,229
scrub,126
peak,96
wetland,89
beach,69
grassland,53

```
Here is another query I was curios about on a personal level. In addition to its reputation as a financial city, San Francisco is sometimes known as a 'green city'. This mostly refers to their policies on energy consumption and recycling, but I thought it would be interesting to see the different instances of natural formations in San Francisco. The number of trees probably does not include the trees in 'Golden Gate Park', because it would show up as an amenity in the data. This is another area of the San Franciso Open Street Map data that could use some improving, because this information does not seem very useful or accurate. For example, there are 69 instances of 'beach', but I'm sure there are less that 69 beaches in San Francisco. Perhaps the Open Street Map data encompasses more than just the city of San Francisco, but this should be more clear. 


# Conclusion and Ideas for Further Study
**********************************************************************************************************************
It seems apparent that the San Francisco Open Street Map data is incomplete. There were several inconsistencies in the data, and there seems to be data out right missing. For example, where did Wells Fargo go? There are certaintly Wells Fargo banks in San Francisco, so its hard to know what the problem is. There was also several instances of foreign language characters, such as Kanji and some arabic text. I even spotted some kind of emoji-like icon. Lastly, it would be interesting to compare these results to other large metros. For example, it might be interesting to know if there are more schools in New York than in San Francisco (according to osm data), both nominally and in terms of percent.

#### Pros vs Cons of chosen cleaning method:

While the cleaning methods do provide more uniformity within the data, the original integrety of the data is arguably lost. For example, the reason why some street names end in 'St', while others end in 'Street' is likely due to the way they are marked on street signs. Calling "Fremont St" by the name of "Fremont Street" is not accurate, because there could also be a "Fremont Street" that refers to a different street in the same city. Also, many rows that contained special characters where dropped, resulting in a significant loss of data. For the purposes of this presentation, however, I believe uniformity is more important that sheer robustness.

#### Works cited and acknowledgements

* Much of the code used to create csv files was provided by Udacity's DAND course. It was modified from the Open Street Maps case study. So a huge thanks to everyone at Udacity.

* As for the code written by myself, much of it was inspired by posts and advice from the DAND forums. Thank you to everyone who has helped me in the forums.

* Thank you to openstreetmap.org and mapzen.com. Open data is one of the greatest concepts in the world.
