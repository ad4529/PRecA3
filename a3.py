import xml.etree.ElementTree as ET
import glob
import os
import numpy as np
import tqdm
import pickle


def write_lg(ui2lginfo, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for ui in ui2lginfo.keys():
        with open(os.path.join(dir_name, ui+'.lg'), 'w') as f:
            for idx, line in enumerate(ui2lginfo[ui]):
                f.write('O, ' + 'sym_' + str(idx) + ', ' + line[0] + ', 1.0, ')
                for s_id in line[1][:-1]:
                    f.write(str(s_id)+', ')
                f.write(str(line[1][-1]) + '\n')
        f.close()

def extract_traces(data_root):

    symbol_files = sorted(glob.glob(os.path.join(data_root, '*', '*.inkml')))

    ui2strokeid2trace = {}
    ui2clslines = {}
    ui2id2cls = {}
    # Get the traces
    print('Parsing the .inkml files..')
    for file in tqdm.tqdm(symbol_files):
        try:
            tree = ET.parse(file)
        except ET.ParseError:
            continue
        root = tree.getroot()

        # Find the UI
        ui = file.split('/')[-1].replace('.inkml', '')

        # Find the strokes with their id
        traces = {}
        for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):
            tr_id = int(trace.attrib['id'])
            points = trace.text.replace('\n', '').split(',')
            points = [pt.lstrip(' ') for pt in points]
            points = [pt.split(' ') for pt in points]
            # Make sure only the x and y are recorded
            points = [[float(p[0]), float(p[1])] for p in points]
            points = np.array(points, dtype=np.float)
            traces[tr_id] = points
        ui2strokeid2trace[ui] = traces

        # Find the strokes with their class
        lines = []
        id2cls = {}
        trace_grps = root.findall('{http://www.w3.org/2003/InkML}traceGroup')[0]
        for trace_grp in trace_grps.findall('{http://www.w3.org/2003/InkML}traceGroup'):
            line = []
            cls_info = trace_grp.find('{http://www.w3.org/2003/InkML}annotation')
            cls = cls_info.text
            if cls == ',':
                cls = 'COMMA'
            line.append(cls)
            for strokeid in trace_grp.findall('{http://www.w3.org/2003/InkML}traceView'):
                stroke = int(strokeid.attrib['traceDataRef'])
                line.append(stroke)
                id2cls[stroke] = cls
            lines.append(lines)
        ui2clslines[ui] = lines
        ui2id2cls[ui] = id2cls

    return ui2strokeid2trace, ui2clslines, ui2id2cls


def s_oracle(ui2id2cls):
    print('Evaluating s-oracle...')
    ui2lginfo = {}
    for ui in tqdm.tqdm(ui2id2cls.keys()):
        lines = []
        for idx in ui2id2cls[ui].keys():
            lines.append([ui2id2cls[ui][idx], [idx]])
        ui2lginfo[ui] = lines

    return ui2lginfo


def k_means(ui2id2tr):
    ui2k2ids = {}
    print('Computing K-Means for the entire dataset...')
    for ui in tqdm.tqdm(ui2id2tr.keys()):
        id2mean = {}
        for idx in ui2id2tr[ui].keys():
            mean = np.mean(ui2id2tr[ui][idx], axis=0)
            id2mean[idx] = mean

        # Run K-Means from 2 to (N-1)-Clusters (1-Means and N-Means are assumed)
        k2ids = {}
        k2ids[1] = [list(id2mean.keys())]
        k2ids[len(id2mean.keys())] = [[idx] for idx in id2mean.keys()]
        for k in range(2, len(id2mean.keys())):

            # Pick cluster means as 1 of the stroke means equally spaced
            ids = sorted(list(id2mean.keys()))
            choices = np.linspace(ids[0], ids[-1], k).astype(np.int)
            choice_ids = [ids[choice] for choice in choices]
            clusters = []
            for choice in choice_ids:
                clusters.append(id2mean[choice])
            clusters = np.asarray(clusters)
            ids2cluster = [[] for _ in range(k)]
            old_clusters = np.zeros(clusters.shape)
            while not np.array_equal(clusters, old_clusters):
                ids2cluster = [[] for _ in range(k)]
                for idx in id2mean.keys():
                    dist = []
                    for cluster in clusters:
                        dist.append(np.linalg.norm(id2mean[idx] - cluster))
                    min_dist = np.argmin(dist)
                    ids2cluster[min_dist].append(idx)

                # Find new cluster centers
                new_cs = []
                for cluster in ids2cluster:
                    pts = []
                    # Handle case where cluster has no points
                    if len(cluster) == 0:
                        break
                    for idx in cluster:
                        pts.append(id2mean[idx])
                    pts = np.asarray(pts)
                    new_c = np.mean(pts, axis=0)
                    new_cs.append(new_c)

                # New cluster centers
                old_clusters = clusters
                clusters = np.asarray(new_cs)
            k2ids[k] = ids2cluster
        ui2k2ids[ui] = k2ids

    with open('K-Means_op.pkl', 'wb') as f:
        pickle.dump(ui2k2ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ui2k2ids


def oracle_op(ui2k2ids, ui2clslines, or_type='K-Oracle_op'):
    print('Choosing the best cluster for {}'.format(or_type))
    ui2k = {}
    for ui in tqdm.tqdm(ui2k2ids.keys()):
        scores = []
        cls_lines = ui2clslines[ui]
        num_elem = sorted([len(line)-1 for line in cls_lines])
        for k in sorted(list(ui2k2ids[ui].keys())):
            score = 0
            if k == len(num_elem):
                score += 1
            if sorted([len(cluster) for cluster in ui2k2ids[ui][k]]) == num_elem:
                score += 1
            # Cluster list for the K
            cluster_ids_score = 0
            for clusters in ui2k2ids[ui][k]:
                for line in cls_lines:
                    if sorted(clusters) == sorted(line):
                        cluster_ids_score += 1
            score += cluster_ids_score/len(cls_lines)
            scores.append(score)

        # Find the chosen k
        ui2k[ui] = np.argmax(scores) + 1

    with open(or_type+'.pkl', 'wb') as f:
        pickle.dump(ui2k, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ui2k


def curate_oracles(ui2k2ids, ui2k, ui2id2cls, type='K-Oracle'):
    print('Curating {} O/P for creating .lg files'.format(type))
    ui2lginfo = {}
    for ui in tqdm.tqdm(ui2k2ids.keys()):
        lines = []
        chosen_k = ui2k2ids[ui][ui2k[ui]]
        for cluster in chosen_k:
            try:
                cls = ui2id2cls[ui][cluster[0]]
            except KeyError:
                cls = '_'
            lines.append([cls, cluster])
        ui2lginfo[ui] = lines
    return ui2lginfo


def ac(ui2id2tr):

    print('Computing Agglomerative clusters...')
    ui2lvl2ids = {}
    for ui in tqdm.tqdm(ui2id2tr.keys()):
        id2mean = {}
        for tr_id in ui2id2tr[ui].keys():
            mean = np.mean(ui2id2tr[ui][tr_id], axis=0)
            id2mean[tr_id] = mean

        # Calculate the pair distances
        trace_ids = sorted(list(ui2id2tr[ui].keys()))
        pair_dists = {}
        for i in range(len(trace_ids)):
            for j in range(i+1, len(trace_ids)):
                pair_dists[tuple((trace_ids[i],trace_ids[j]))] = np.linalg.norm(
                    id2mean[trace_ids[i]] - id2mean[trace_ids[j]])

        n_clusters = list(id2mean.keys())
        n_clusters = [[idx] for idx in n_clusters]
        tot_clusters = len(n_clusters)
        # The first level will be all traces separate
        lvl2ids = {tot_clusters: n_clusters.copy()}

        while tot_clusters > 1:
            min_dist = 10000
            min_dist_clusters = tuple((0,1))
            for i in range(len(n_clusters)):
                for j in range(i+1, len(n_clusters)):
                    for tr_id_a in n_clusters[i]:
                        for tr_id_b in n_clusters[j]:
                            dist = pair_dists[tuple(sorted(list([tr_id_a, tr_id_b])))]
                            if dist < min_dist:
                                min_dist = dist
                                min_dist_clusters = tuple((i,j))

            # Now merge the min_dist clusters
            new_clusters = []
            for i in range(len(n_clusters)):
                if i != min_dist_clusters[1]:
                    new_clusters.append(n_clusters[i].copy())
                else:
                    new_clusters[min_dist_clusters[0]] += n_clusters[i].copy()

            tot_clusters = len(new_clusters)
            n_clusters = new_clusters
            lvl2ids[tot_clusters] = n_clusters
        ui2lvl2ids[ui] = lvl2ids

    with open('ac_op.pkl', 'wb') as f:
        pickle.dump(ui2lvl2ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ui2lvl2ids


def main():
    ui2id2tr, ui2clslines, ui2id2cls = extract_traces('data/data/inkml')
    # ui2lginfo = s_oracle(ui2id2cls)
    # if not os.path.exists('K-Means_op.pkl'):
    #     ui2k2ids = k_means(ui2id2tr)
    # else:
    #     with open('K-Means_op.pkl', 'rb') as f:
    #         ui2k2ids = pickle.load(f)
    # print(ui2k2ids['65_alfonso'][5])
    # if not os.path.exists('K-Oracle_op.pkl'):
    #     ui2k = k_oracle(ui2k2ids, ui2clslines)
    #
    # else:
    #     with open('K-Oracle_op.pkl', 'rb') as f:
    #         ui2k = pickle.load(f)
    # ui2lginfo_k_oracle = curate_k_oracle(ui2k2ids, ui2k, ui2id2cls)
    # write_lg(ui2lginfo_k_oracle, 'K-Oracle')
    if not os.path.exists('ac_op.pkl'):
        ui2lvl2ids = ac(ui2id2tr)
    else:
        with open('ac_op.pkl', 'rb') as f:
            ui2lvl2ids = pickle.load(f)

    if not os.path.exists('AC-Oracle_op.pkl'):
        ui2c = oracle_op(ui2lvl2ids, ui2clslines, or_type='AC-Oracle_op')
    else:
        with open('AC-Oracle_op.pkl', 'rb') as f:
            ui2c = pickle.load(f)
    ui2lginfo_ac_oracle = curate_oracles(ui2lvl2ids, ui2c, ui2id2cls, type='AC-Oracle')
    write_lg(ui2lginfo_ac_oracle, 'AC-Oracle')


if __name__ == '__main__':
    main()
