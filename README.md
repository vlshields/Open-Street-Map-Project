# Open-Street-Map-Project
My project submission for the data cleaning/wrangling section of the DAND course from Udacity

1) The original osm file was first audited with the audit.py file. To test the code, set the value OSMFILE to the name of the sample osm file provided.

2) After the osm file was audited, the data was cleaned and formatted to csv format. To test this code, set the value OSM_PATH to the name of the sample osm file provided. You can set validate=False if the code is running too slowly. As Udacity mentions in the last excersize open street maps case study, validation makes the code run about 10x more slowly.

3) I ran into an issue where some of the values in element.attrib where missing in the nodes. Specifically, there were a few nodes missing the values 'user' and 'uid'. I know it probably wasn't the best way to handle the problem, but there where only four nodes that had this issue.

4) The file 'splitting_osm.py' was provided by Udacity, and was used to create the sample osm file provided. You don't need to do anything with this code, but I included it for reference.

Other notes:
- The file 'schema.py' is provided and is required in the same directory

- The 'schema.py' file should be identical to the schema provided in the Open Street Maps case study

-Although it doesn't say so in the project submission instructions, I thought it would be wise to provide the database in which the SQL queries seen in the project file where ran.
