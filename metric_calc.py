import os
import subprocess
import csv

sourcemeter_java_exe = r"S:/SourceMeter-10.2.0-x64-Windows/Java/AnalyzerJava.exe"
clone_dir = r"repositories"

if not os.path.exists("repositories"):
    os.makedirs("repositories")

with open("urls.txt", "r", encoding='utf-8') as f:
    urls = f.read().splitlines()


with open("results.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(
        [
            "Repository",
            "URL",
            "LOC",
            "NOC",
            "NCL",
            "RFC",
            "CBO",
            "DIT",
            "LCOM5",
            "WMC"
        ]
    )

    i = 1
    for url in urls:
        try:
            repo_name = url.split("/")[-1]

            clone_path = os.path.join(clone_dir, str(i), repo_name)
            if not os.path.exists(clone_path):
                clone_cmd = f"git clone {url} {clone_path}"
                subprocess.run(clone_cmd, shell=True, check=True)

            project_name = repo_name
            project_base_dir = clone_path
            results_dir = os.path.join(clone_dir, "SourceMeter_Results", str(i), repo_name)
            os.makedirs(results_dir, exist_ok=True)

            analysis_cmd = f"{sourcemeter_java_exe} -projectName={project_name} -projectBaseDir={project_base_dir} -resultsDir={results_dir}"
            subprocess.run(analysis_cmd, shell=True, check=True)

            class_csv_file_name = f"{project_name}-Class.csv"
            package_csv_file_name = f"{project_name}-Package.csv"

            total_loc = 0
            total_noc = 0
            total_ncl = 0
            total_rfc = 0
            total_cbo = 0
            total_dit = 0
            total_lcom5 = 0
            total_wmc = 0
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file == class_csv_file_name:
                        class_csv_file = os.path.join(root, file)

                        with open(class_csv_file, "r") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                total_loc += int(row["LOC"])
                                total_noc += int(row["NOC"])
                                total_rfc += int(row["RFC"])
                                total_cbo += int(row["CBO"])
                                total_dit += int(row["DIT"])
                                total_lcom5 += int(row["LCOM5"])
                                total_wmc += int(row["WMC"])

                    if file == package_csv_file_name:
                        package_csv_file = os.path.join(root, file)

                        with open(package_csv_file, "r") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                total_ncl += int(row["NCL"])

            writer.writerow(
                [
                    repo_name,
                    url,
                    total_loc,
                    total_noc,
                    total_ncl,
                    total_rfc,
                    total_cbo,
                    total_dit,
                    total_lcom5,
                    total_wmc
                ]
            )
        except Exception as e:
            print(f"Error in {repo_name}: {e}")

        i += 1
