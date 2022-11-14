#!/usr/bin/env python

import click
from functools import partial
from fastcluster import linkage
import json
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from scipy.cluster.hierarchy import fcluster
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import subprocess as sp
import time
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"
import warnings
warnings.filterwarnings("ignore")

from explainn.interpretation.interpretation import pwm_to_meme
from pwm_scoring import _get_PWMs
from utils import get_file_handle

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "meme_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-c", "--cpu-threads",
    help="Number of CPU threads to use.",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--clustering",
    help="Clustering approach.",
    type=click.Choice(["complete", "iterative"], case_sensitive=False),
    default="complete",
    show_default=True,
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True,
)
@click.option(
    "--tomtom-dir",
    help="Tomtom directory.",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-t", "--time",
    help="Return the program's running execution time in seconds.",
    is_flag=True,
)

def main(**args):

    # Start execution
    start_time = time.time()

    # Initialize
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Save exec. parameters as JSON
    json_file = os.path.join(args["output_dir"],
                             f"parameters-{os.path.basename(__file__)}.json")
    handle = get_file_handle(json_file, "wt")
    handle.write(json.dumps(args, indent=4, sort_keys=True))
    handle.close()

    # Create output dirs
    motifs_dir = os.path.join(args["output_dir"], "motifs")
    if not os.path.isdir(motifs_dir):
        os.makedirs(motifs_dir)
    clusters_dir = os.path.join(args["output_dir"], "clusters")
    if not os.path.isdir(clusters_dir):
        os.makedirs(clusters_dir)
    if not args["tomtom_dir"]:
        tomtom_dir = os.path.join(args["output_dir"], "tomtom")
    else:
        tomtom_dir = args["tomtom_dir"]
    if not os.path.isdir(tomtom_dir):
        os.makedirs(tomtom_dir)   

    # Get motifs
    global motifs
    motifs = []
    global names_int
    names_int = {}
    global names_str
    names_str = {}
    int_names = {}
    _get_motifs(args["meme_file"], motifs_dir, args["cpu_threads"])
    for m in os.listdir(motifs_dir):
        motifs.append(os.path.join(motifs_dir, m))
        _, n = _get_PWMs(motifs[-1])
        base_name = os.path.splitext(os.path.basename(m))[0]
        names_int.setdefault(n[0], int(base_name))
        names_str.setdefault(n[0], base_name)
        int_names.setdefault(int(base_name), n[0])

    # Compute Tomtom similarities
    npz_file = os.path.join(args["output_dir"], "similarities.npz")
    if not os.path.exists(npz_file):
        kwargs = {"bar_format": bar_format, "total": len(motifs)}
        pool = Pool(args["cpu_threads"])
        p = partial(_compute_Tomtom_similarities, meme_file=args["meme_file"],
                    tomtom_dir=tomtom_dir)
        for _ in tqdm(pool.imap(p, motifs), **kwargs):
            pass
        arr = _load_Tomtom_similarities(tomtom_dir, args["cpu_threads"])
        np.savez_compressed(npz_file, similarities=arr)
    # arr = np.load(npz_file)["similarities"]

    # Get clusters
    tsv_file = os.path.join(args["output_dir"], "clusters.tsv.gz")
    if not os.path.exists(tsv_file):   
        if args["clustering"].lower() == "complete":
            Z = linkage(arr, method="complete", metric="correlation")
            cl = fcluster(Z, 0.7, criterion="distance")
            counts = np.bincount(cl)
            singletons = np.where(counts==1)[0]
            nonsingletons = np.where(counts>1)[0]
            cl[np.isin(cl, singletons)] = -1
            for i, cluster in enumerate(nonsingletons):
                cl[np.where(cl == cluster)] = i + 1
        elif args["clustering"].lower() == "iterative":
            # i.e., my way
            cluster = 1
            threshold = -np.log10(0.5)
            done_arr = np.array([], dtype=int)
            done_set = set()
            cl = -np.ones(arr.shape[0], dtype=int)
            row_sums = arr.sum(axis=1)
            idx_sorted = np.argsort(row_sums)[::-1]
            kwargs = {"bar_format": bar_format, "total": len(idx_sorted)}
            for idx in tqdm(idx_sorted, **kwargs):
                if idx in done_set:
                    continue
                # i.e., focus on similar motifs, ignore the rest
                idx_similar_motifs = np.where(arr[idx, :] >= threshold)[0]
                m = np.in1d(idx_similar_motifs, done_arr)
                idx_similar_motifs = idx_similar_motifs[~m]
                if len(idx_similar_motifs) > 1:
                    X = arr[idx_similar_motifs, :]
                    Z = linkage(X, method="complete", metric="correlation")
                    cl_similar_motifs = fcluster(Z, 0.7, criterion="distance")
                    d = dict(list(zip(idx_similar_motifs, cl_similar_motifs)))
                    idx_same_cl = idx_similar_motifs[
                        np.where(cl_similar_motifs == d[idx])[0]
                    ]
                    if len(idx_same_cl) > 1:
                        cl[idx_same_cl] = cluster
                        cluster += 1
                    done_arr = np.append(done_arr, idx_same_cl)
                    done_set.update(idx_same_cl)
                else:
                    done_arr = np.append(done_arr, [idx])
                    done_set.add(idx)
        names = [k for k, _ in sorted(names_int.items(), key=lambda x: x[1])]
        clusters = pd.DataFrame({"Motif_ID": names, "Cluster": cl})
        counts = clusters["Cluster"].value_counts()
        cl = counts.index[np.where(counts==1)]
        clusters.loc[clusters["Cluster"].isin(cl), "Cluster"] = -1
        clusters.to_csv(tsv_file, sep="\t", index=False)
    clusters = pd.read_table(tsv_file)

    # Process clusters
    kwargs = {"bar_format": bar_format,
            "total": len(clusters["Cluster"].unique())}
    pool = Pool(args["cpu_threads"])   
    p = partial(_process_cluster, clusters=clusters, clusters_dir=clusters_dir,
                tomtom_dir=tomtom_dir, output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, clusters["Cluster"].unique()), **kwargs):
        pass
    meme_file = os.path.join(args["output_dir"], "clusters.meme")
    if not os.path.exists(meme_file):
        pwms = []
        names = []
        for clusters_file in os.listdir(clusters_dir):
            if not clusters_file.endswith(".meme"):
                continue
            if clusters_file.startswith("unclust."):
                continue
            pwm, name = _get_PWMs(os.path.join(clusters_dir, clusters_file))
            pwms.append(pwm[0])
            names.append(name[0])
        (names, pwms) = zip(*sorted(zip(names, pwms)))
        pwm_to_meme(pwms, meme_file, dict(list(enumerate(names))))

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Execution time {seconds} seconds")

def _get_motifs(meme_file, motifs_dir, cpu_threads=1):

    # Initialize
    motifs = []
    parse = False

    # Get motifs
    handle = get_file_handle(meme_file, "rt")
    for line in handle:
        line = line.strip("\n")
        if line.startswith("MOTIF"):
            motifs.append([])
            parse = True
        if parse:
            motifs[-1].append(line)
    handle.close()

    # Create motif files
    zfill = len(str(len(motifs)))
    kwargs = {"bar_format": bar_format, "total": len(motifs)}
    pool = Pool(cpu_threads)
    p = partial(__write_motif, motifs_dir=motifs_dir, zfill=zfill)
    for _ in tqdm(pool.imap(p, enumerate(motifs)), **kwargs):
        pass

def __write_motif(i_motif, motifs_dir, zfill=0):

    # Initialize
    i, motif = i_motif
    prefix = str(i).zfill(zfill)

    motif_file = os.path.join(motifs_dir, f"{prefix}.meme")
    if not os.path.exists(motif_file):
        handle = get_file_handle(motif_file, "wt")
        handle.write("MEME version 4\n\n")
        handle.write("ALPHABET= ACGT\n\n")
        handle.write("strands: + -\n\n")
        handle.write(
            "Background letter frequencies (from uniform background):\n"
        )
        handle.write("A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")
        for line in motif:
            handle.write(f"{line}\n")
        handle.close()

def _compute_Tomtom_similarities(motif_file, meme_file, tomtom_dir):

    # Initialize
    prefix = os.path.splitext(os.path.basename(motif_file))[0]
    tomtom_file = os.path.join(tomtom_dir, f"{prefix}.tsv.gz")

    if not os.path.exists(tomtom_file):

        # Compute motif similarities
        cmd = ["tomtom", "-dist", "kullback", "-motif-pseudo", str(0.1),
               "-text", "-min-overlap", str(1), "-thresh", str(1), motif_file,
               meme_file]
        proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)

        # Save Tomtom results
        handle = get_file_handle(tomtom_file, "wb")
        for line in proc.stdout.decode().split("\n"):
            handle.write(f"{line}\n".encode())
        handle.close()

def _load_Tomtom_similarities(tomtom_dir, cpu_threads=1):

    # Initialize
    files = list(os.path.join(tomtom_dir, f) for f in os.listdir(tomtom_dir) \
                 if f.endswith(".tsv.gz"))
    arr = np.empty((len(files), len(files)), dtype=np.half)

    # Load Tomtom files
    kwargs = {"bar_format": bar_format, "total": len(files)}
    pool = Pool(cpu_threads)
    for (idx, evalues) in tqdm(
        pool.imap(__load_Tomtom_similarities, files), **kwargs
    ):
        arr[idx, :] = evalues
        arr[:, idx] = evalues

    return(arr)

def __load_Tomtom_similarities(tomtom_file):

    # Initialize
    global names_int
    col_names = ["Query_ID", "Target_ID", "E-value"]

    df = pd.read_table(tomtom_file, header=0, usecols=col_names, comment="#")
    query_ids = [names_int[n] for n in df["Query_ID"].tolist()]
    df["Query_ID"] = np.array(query_ids).astype(int)
    target_ids = [names_int[n] for n in df["Target_ID"].tolist()]
    df["Target_ID"] = np.array(target_ids).astype(int)
    df.sort_values(by=["Target_ID"], inplace=True)
    evalues = df["E-value"].to_numpy()
    evalues = -np.log10(evalues)
    evalues[np.where(evalues > 10)] = 10
    evalues[np.where(evalues < -2)] = -2

    return(query_ids[0], evalues.astype(np.half))

###############################################################################
# https://github.com/jvierstra/motif-clustering/Workflow_v2.1beta-human.ipynb #
###############################################################################

def _abs_mean(x):
    return(np.mean(np.abs(x)))

def _process_cluster(cluster_id, clusters, clusters_dir, tomtom_dir,
                     output_dir="./"):

    # Initialize
    global names_str
    cluster_meme_file = os.path.join(clusters_dir, f"{cluster_id}.meme")

    if os.path.exists(cluster_meme_file):
        return

    cluster = clusters.groupby("Cluster").get_group(cluster_id)

    # Save cluster
    if cluster_id == -1:
        cluster_file = os.path.join(clusters_dir, "unclust.tsv.gz")
    else:
        cluster_file = os.path.join(clusters_dir, f"{cluster_id}.tsv.gz")
    if not os.path.exists(cluster_file):
        cluster.to_csv(cluster_file, sep="\t", index=False)

    # Get motifs
    motifs = cluster["Motif_ID"].to_list()
    if cluster_id == -1:

        pwms = []
        names = []

        for motif_id in motifs:
            meme_file = os.path.join(output_dir, "motifs",
                                     f"{names_str[motif_id]}.meme")
            pwm, name = _get_PWMs(meme_file)
            pwms.append(pwm[0])
            names.append(name[0])

        meme_file = os.path.join(clusters_dir, "unclust.meme")
        pwm_to_meme(pwms, meme_file, names, verbose=False)

    else:

        tomtom = __load_Tomtom_files(motifs, tomtom_dir)

        rows = tomtom["Query_ID"].isin(motifs) & \
            tomtom["Target_ID"].isin(motifs)
        pairwise = tomtom[rows]

        seed_motif = pairwise.groupby("Query_ID")\
                            .agg({"Optimal_offset": _abs_mean})\
                            .sort_values("Optimal_offset").index[0]
        rows = (tomtom["Query_ID"] == seed_motif) & \
                tomtom["Target_ID"].isin(motifs)
        pairwise = tomtom[rows]

        w = pairwise["Target_consensus"].str.len()
        left = min(-pairwise["Optimal_offset"])
        l_offset = -left - pairwise["Optimal_offset"]
        right = max(l_offset + w)
        r_offset = right - w - l_offset

        alignment = pairwise.drop(["Query_ID", "Optimal_offset"], axis=1)
        alignment.loc[:, "w"] = w
        alignment.loc[:, "l_offset"] = l_offset
        alignment.loc[:, "r_offset"] = r_offset
        alignment.columns = ["motif", "consensus", "strand", "w",
                                "l_offset", "r_offset"]

        alignment.reset_index(drop=True, inplace=True)

        n = len(alignment)
        l = min(alignment["l_offset"])
        r = max(alignment["r_offset"] + alignment["w"])
        w = r - l

        summed_pwm = np.zeros((4, w))

        for _, row in alignment.iterrows():

            motif_id = row["motif"]
            reverse_complement = row["strand"] == "-"
            left = row["l_offset"]
            width = row["w"]

            meme_file = os.path.join(output_dir, "motifs",
                                    f"{names_str[motif_id]}.meme")
            pwm, _ = _get_PWMs(meme_file)
            pwm = pwm[0]
            if reverse_complement:
                pwm = pwm[::-1,::-1]

            extended_pwm = np.ones((4, w)) * 0.25
            extended_pwm[:,left:left+width] = pwm

            summed_pwm += extended_pwm

        avg_pwm = (summed_pwm / n)

        # Save archetype motif
        pwm_to_meme(np.expand_dims(avg_pwm, axis=0), cluster_meme_file,
                    dict({0: f"cluster{cluster_id}"}), verbose=False)

def __load_Tomtom_files(motifs, tomtom_dir):

    # Initialize
    global names_str
    dfs = []
    col_names = ["Query_ID", "Target_ID", "Optimal_offset", "Target_consensus",
                 "Orientation"]  

    for motif in motifs:
        f = os.path.join(tomtom_dir, f"{names_str[motif]}.tsv.gz")
        dfs.append(pd.read_table(f, header=0, usecols=col_names, comment="#"))

    return(pd.concat(dfs))

if __name__ == "__main__":
    main()