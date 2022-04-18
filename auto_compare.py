import networkx as nx
import matplotlib.pyplot as plt
import os
import csv
import subprocess

from outcome import capture

part_path = "partitions"
radatools_path = os.path.join("radatools-5.2-win32", "Communities_Tools", "Compare_Partitions.exe")

# First, create the general csv file to which we will append all the data from each comparaisons
gather_csv_path = os.path.join("comparaison_metrics.csv")
gather_csv_file = open(gather_csv_path, "w")

gather_csv_file.write(",".join(["Reference Name",
                        "Compared Name",
                        "Normalized Mutual Information Index (arithmetic)",
                        "Normalized Variation Of Information Metric",
                        "Jaccard Index"]) + "\n")

for r, s, files in os.walk(part_path):
    if len(files):
        print(f"------- Network: {r} -----")
        ref_clu = [f for f in files if not f.split("_")[-1].split(".")[0] in ["asyn", "girvan-newman", "greedy", "louvain", "spinglass", "walktrap"] and f.split(".")[-1] == "clu"]
        to_compare_clu = [f for f in files if not f in ref_clu and f.split(".")[-1] == "clu"]

        print("Reference partitions: ", ref_clu)
        print("To be compared paritions: ", to_compare_clu)

        # Compare each to_compare_clu files to references clu files



        # Then, compare each non-ref clu files to references partitions
        for i, ref in enumerate(ref_clu):
            for f in to_compare_clu:
                print(f"Comparing {f} to {ref}")
                temp_csv_path = os.path.join(r, ref.split(".")[0] + ".csv")
                subprocess.run([radatools_path,
                                    os.path.join(r, f),
                                    os.path.join(r, ref),
                                    temp_csv_path,
                                    "T"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)

                print(f"Appending {temp_csv_path} to {gather_csv_path}")

                temp_csv_file = open(temp_csv_path, "r")
                temp_csv_reader = csv.DictReader(temp_csv_file, delimiter="\t")

                for row in temp_csv_reader:
                    gather_csv_file.write(",".join([ref,
                                                f,
                                                row["Normalized_Variation_Of_Information_Metric"],
                                                row["Normalized_Mutual_Information_Index_(arithmetic)"],
                                                row["Jaccard_Index"]]) + "\n")

                temp_csv_file.close()
                os.remove(temp_csv_path)
        """
        if len(ref_clu):
            # Filtering comparaison csv's to a new csv
            general_csv_path = os.path.join(r, ref_clu[0].split(".")[0] + "_comp.csv")
            with open(general_csv_path, "w") as csvfile:
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

                        print(f"Reading {to_insert_csv_path} for append to {general_csv_path}...")

                        to_insert_csvreader = csv.DictReader(to_insert_csv_file, delimiter="\t")

                        for row in to_insert_csvreader:
                            #print("row", row)
                            to_write = [ref, f, row["Normalized_Variation_Of_Information_Metric"],
                                                row["Normalized_Mutual_Information_Index_(arithmetic)"],
                                                row["Jaccard_Index"]]

                            #print(to_write)
                            csw.writerow(to_write)

                        to_insert_csv_file.close()
    """