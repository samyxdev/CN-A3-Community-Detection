import networkx as nx
import matplotlib.pyplot as plt
import os
import csv
import subprocess

from outcome import capture

part_path = "partitions"
radatools_path = os.path.join("radatools-5.2-win32", "Communities_Tools", "Compare_Partitions.exe")

for r, s, files in os.walk(part_path):
    if len(files):
        print(f"------- Network: {r} -----")
        ref_clu = [f for f in files if not f.split("_")[-1].split(".")[0] in ["asyn", "girvan-newman", "greedy", "louvain", "spinglass", "walktrap"] and f.split(".")[-1] == "clu"]
        to_compare_clu = [f for f in files if not f in ref_clu and f.split(".")[-1] == "clu"]

        print("Reference partitions: ", ref_clu)
        print("To be compared paritions: ", to_compare_clu)

        # Compare each to_compare_clu files to references clu files
        for i, ref in enumerate(ref_clu):
            for f in to_compare_clu:
                print(f"Comparing {f} to {ref}")
                csv_path = os.path.join(r, ref.split(".")[0] + ".csv")
                subprocess.run([radatools_path,
                                    os.path.join(r, f),
                                    os.path.join(r, ref),
                                    csv_path,
                                    "T"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)


        if len(ref_clu):
            # Filtering comparaison csv's to a new csv
            with open(os.path.join(r, ref_clu[0].split(".")[0] + "_comp.csv"), "w") as csvfile:
                csw = csv.writer(csvfile)
                csw.writerow(["Reference Name",
                                "Compared Name",
                                "Normalized Mutual Information Index (arithmetic)",
                                "Normalized Variation Of Information Metric",
                                "Jaccard Index"])

                for ref in ref_clu:
                    for f in to_compare_clu:
                        to_insert_csv_path = os.path.join(r, ref.split(".")[0] + ".csv")
                        to_insert_csv_file = open(to_insert_csv_path, "r")

                        to_insert_csvreader = csv.DictReader(to_insert_csv_file, delimiter="\t")

                        for row in to_insert_csvreader:
                            #print("row", row)
                            to_write = [ref, f, row["Normalized_Variation_Of_Information_Metric"],
                                                row["Normalized_Mutual_Information_Index_(arithmetic)"],
                                                row["Jaccard_Index"]]

                            #print(to_write)
                            csw.writerow(to_write)

                        to_insert_csv_file.close()
