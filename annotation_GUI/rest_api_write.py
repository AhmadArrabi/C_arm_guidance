import requests
import random

mysql_url = '<DATABASE-LINK>'

def generate_random_float():
    return round(random.uniform(1.0, 100.0), 2)
def send_post(x,y,z,a,b,order,sample_name,group_id,target):
# Function to generate random float within a specified range
    
    data = {
        'x': x,
        'y': y,
        'z': z,
        'a': a,
        'b': b,
        'order' : order,
        'sample_name': sample_name,
        'group_id': group_id,
        'target': target
    }
    # Make a POST request to insert data into the MySQL table
    response = requests.post(mysql_url, data=data)