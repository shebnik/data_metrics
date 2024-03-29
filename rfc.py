import os
import subprocess
import csv

sourcemeter_java_exe = r"D:/SourceMeter/Java/AnalyzerJava.exe"
clone_dir = r"repositories"

if not os.path.exists("repositories"):
    os.makedirs("repositories")

# Read GitHub URLs from urls.txt
with open("urls.txt", "r") as f:
    urls = f.read().splitlines()


with open("results.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["Repository", "URL", "Total RFC", "Total CBO"])

    # Loop through each URL
    i = 1
    for url in urls:
        # Extract repository name from URL
        repo_name = url.split("/")[-1]

        # Clone the repository
        clone_path = os.path.join(clone_dir, str(i), repo_name)
        clone_cmd = f"git clone {url} {clone_path}"
        subprocess.run(clone_cmd, shell=True, check=True)

        # Run SourceMeter Java analysis
        project_name = repo_name
        project_base_dir = clone_path
        results_dir = os.path.join(clone_dir, "SourceMeter_Results", str(i), repo_name)
        os.makedirs(results_dir, exist_ok=True)

        analysis_cmd = f"{sourcemeter_java_exe} -projectName={project_name} -projectBaseDir={project_base_dir} -resultsDir={results_dir} -runFB=true -FBFileList=filelist.txt"
        subprocess.run(analysis_cmd, shell=True, check=True)

        print(f"Analysis completed for {repo_name}")

        # Extract total RFC and CBO metrics and write to results.csv
        class_csv_file_name = f"{project_name}-Class.csv"

        total_rfc = 0
        total_cbo = 0
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file == class_csv_file_name:
                    class_csv_file = os.path.join(root, file)

                    with open(class_csv_file, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            total_rfc += int(row["RFC"])
                            total_cbo += int(row["CBO"])

        writer.writerow([repo_name, url, total_rfc, total_cbo])
        print(f"Results written for {[repo_name, url, total_rfc, total_cbo]}")

        i += 1
