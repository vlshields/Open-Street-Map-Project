import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "sample.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road",
            "Trail", "Parkway", "Commons", 'Way'] # some additional names common in SF

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name) #search for the text pattern
    if m:
        street_type = m.group() #If match is found, extract street name
        if street_type not in expected:
            street_types[street_type].add(street_name) # If street name not in the expected list, add to a set


def is_street_name(elem):
    # Returns True if the return statement is true, False otherwise
    return (elem.attrib['k'] == "addr:street")


def audit(path):
    with open(path) as osm_file:
        street_types = defaultdict(set)
        # iterate throuth the osm file and call the functions above
        for event, elem in ET.iterparse(osm_file, events=("start",)):
            if elem.tag == "node" or elem.tag == "way":
                for tag in elem.iter("tag"):
                    if is_street_name(tag):
                        audit_street_type(street_types, tag.attrib['v'])
    return street_types


street_types = audit(OSMFILE)

pprint.pprint(street_types)



