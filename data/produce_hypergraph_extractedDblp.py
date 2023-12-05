import gzip
import json
from collections import defaultdict, OrderedDict
import numpy as np
import networkx as nx


categories = OrderedDict([
    ('theory', ['stoc', 'soda', 'focs', 'icalp']),
    ('dmw', ['kdd', 'icdm', 'wsdm', 'www']),
    ('ivc', ['cvpr', 'iccv', 'eccv']),
    ('ml', ['nips', 'colt', 'icml']),
    ('idm', ['sigmod', 'pods', 'vldb']),
    ('net', ['sigcomm', 'infocom', 'mobicom', 'imc']),
])

with gzip.GzipFile('dblp.json.gz', 'r') as f:
    papers_raw = json.loads(f.read().decode())

edges = defaultdict(int)
author_areas = defaultdict(lambda: defaultdict(int))
paper_count = 0
weights = defaultdict(int)
authors = set()
papers = set()
paper_categories = dict()
for paper_title, paper_authors, paper_year in papers_raw:
    for cat, (category, conferences) in enumerate(categories.items()):
        for conference in conferences:
            if conference in paper_title.split('/'):
                paper_categories[f'{paper_title}_in'] = cat
                paper_categories[f'{paper_title}_out'] = cat
                papers.add(f'{paper_title}_in')
                papers.add(f'{paper_title}_out')
                edges[(f'{paper_title}_in', f'{paper_title}_out')] = 1
                weights[f'{paper_title}_in'] = 0
                weights[f'{paper_title}_out'] = 0
                for author in paper_authors:
                    authors.add(author)
                    author_areas[author][category] += 1
                    weights[author] += 1
                    edges[(author, f'{paper_title}_in')] = 1
                    edges[(f'{paper_title}_out', author)] = 1
                paper_count += 2


edgelist = [(u, v, w) for (u, v), w in edges.items()]
G = nx.DiGraph()
G.add_weighted_edges_from(edgelist)

extractedDblp = sorted(max(nx.weakly_connected_components(G), key=len))
G = G.subgraph(extractedDblp)
n = len(G)
m = 2 * len(G.edges) - n
authors = sorted(authors.intersection(extractedDblp))
papers = sorted(papers.intersection(extractedDblp))
nodes = {v: i + 1 for i, v in enumerate(authors)}
author_count = len(nodes)
nodes.update({f'{v}': i + 1 + author_count for i, v in enumerate(papers)})
print(len(nodes), max(nodes.values()))
degrees = G.degree()
with open('hyperExtractedDblp.heg2', "w") as f_out:
    print(nx.number_of_nodes(G), nx.number_of_edges(G), "011", file=f_out)
    for a in authors:
        print(weights[a], end='', file=f_out)
        for _, u, w in G.edges(a, data='weight'):
            print(f' {nodes[u]} {w}', end='', file=f_out)
        print(file=f_out)
    for p in papers:
        print(weights[p], end='', file=f_out)
        for _, u, w in G.edges(p, data='weight'):
            print(f' {nodes[u]} {w}', end='', file=f_out)
        print(file=f_out)

with open('hyperExtractedDblp_authorAreas.txt', 'w') as p_out:
    for p in papers:
        print(paper_categories[p], file=p_out)

with open('hyperExtractedDblp.hmetis', 'w') as hmetis_out:
    print(len(papers) // 2, author_count, "10", file=hmetis_out)
    for p in papers:
        if not p.endswith('out'):
            continue
        print(' '.join([str(nodes[a]) for _, a in G.edges(p)]), file=hmetis_out)
    for a in authors:
        print(len(G.edges(a)), file=hmetis_out)

areas_array = np.array([[author_areas[a][cat] for cat in sorted(categories.keys())] for a in authors])
labels = np.argmax(areas_array, axis=1)
with open('hyperExtractedDblp.labels', 'w') as l_out:
    print(*sorted(categories.keys()), file=l_out)
    for label in labels:
        print(label, file=l_out)
# edgelistFile = "dcExtractedDblp.edgelist"
# nx.write_edgelist(G, edgelistFile, data='weight', delimiter='\t')
# areas_df = pd.DataFrame(author_areas, dtype='Int64')[extractedDblp].fillna(0).T
# print(areas_df)
# areas_df.to_csv('dcExtractedDblp_authorAreas.txt', sep=' ')
