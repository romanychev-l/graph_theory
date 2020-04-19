from lxml import etree, objectify
import matplotlib.pyplot as plt
import numpy as np
import csv

FILENAME = "saratov.osm"

data = etree.parse(FILENAME)

with open(FILENAME, 'rb') as f:
    osm = f.read()

root = objectify.fromstring(osm)

road = []
roads = []
print(1)
way = root.way
for row in way:
    tag = row.find('tag')
    if not (tag is None) and tag.attrib['k'] == 'highway':
        nd = row.nd
        road = []
        for ref in nd:
            road.append(int(ref.attrib['ref']))
        roads.append(road)

node = root.node
        
print(len(node), 2)
#nodes = {int(row.attrib['id']): [float(row.attrib['lat']), float(row.attrib['lon'])] for row in node}
nodes = {}
for row in node:
    nodes[int(row.attrib['id'])] = [float(row.attrib['lat']), float(row.attrib['lon'])]
print(2.5)
#count_node_in_roads = {int(row.attrib['id']): 0 for row in node}
count_node_in_roads = {}
for row in node:
    count_node_in_roads[int(row.attrib['id'])] = 0
print(3)
for road in roads:
    for nd in road:
        count_node_in_roads[int(nd)] = count_node_in_roads[int(nd)] + 1
print(4)
count_edge_in_roads = {int(row.attrib['id']): 0 for row in node}
for road in roads:
    for i in range(len(road) - 1):
        count_edge_in_roads[road[i]] = count_edge_in_roads[road[i]] + 1
        count_edge_in_roads[road[i+1]] = count_edge_in_roads[road[i+1]] + 1

nodes_to_delete = []
for nd in nodes:
    if count_node_in_roads[nd] == 0 or (count_node_in_roads[int(nd)] == 1 and count_edge_in_roads[int(nd)] == 2):
        nodes_to_delete.append(nd)
print(len(nodes_to_delete), len(nodes))
for nd in nodes_to_delete:
    nodes.pop(nd)

list_adj = {nd:[] for nd in nodes}

for road in roads:
    row = []
    for nd in road:
        if nd in nodes:
            row.append(nd)
    for i in range(len(row) - 1):
        list_adj[row[i]].append(int(row[i+1]))
        list_adj[row[i+1]].append(int(row[i]))


for key in list_adj:
    list_adj[key] = list(set(list_adj[key]))

line_csv = []
with open("list_saratov.csv", 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for key in list_adj:
        line_csv.append(key)
        for r in list_adj[key]:
            line_csv.append(r)
        writer.writerow(line_csv)
        line_csv = []


fig = plt.gcf()
fig.set_size_inches(20, 24)
Lon_Lat = []
#print(nodes)
for road in roads:
    for index_node_in_road in road:
        if nodes.get(index_node_in_road):
            Lon_Lat.append([nodes.get(index_node_in_road)[1], nodes.get(index_node_in_road)[0]])
    Lon_Lat = np.array(Lon_Lat)
    #print(len(Lon_Lat))
    if len(Lon_Lat) > 0:
        plt.plot(Lon_Lat[::, 0], Lon_Lat[::, 1], 'blue')
    Lon_Lat = []

fig.set_size_inches(20, 24, forward=True)
fig.savefig('saratov_orig.png', dpi=100)
plt.show()

