import os
import re
import csv
import shutil

language = 'java'

def calculate_metrics(url):
    # clone the repository
    repo_name = url.split('/')[-1]
    os.system(f'git clone {url} repositories/{repo_name}')
    print(f'Cloned {repo_name} repository')

    # get all the java files in the repository
    java_files = []
    for root, dirs, files in os.walk(f'repositories/{repo_name}'):
        for file in files:
            if file.endswith(f".{language}"):
                java_files.append(os.path.join(root, file))

    # initialize the metrics variables
    LOC = 0
    NC = 0
    ANA = 0
    ANM = 0
    ANSM = 0
    ANGM = 0
    ANCM = 0
    NGen = 0
    NAssoc = 0

    # loop through each java file
    for file in java_files:
        # read the contents of the file
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f'Error reading file {file}: UnicodeDecodeError')
            continue

        # calculate the LOC metric
        lines = content.split('\n')
        LOC += len(lines)

        # calculate the NC metric
        class_pattern = re.compile(
            r'class\s+([A-Za-z0-9_]+)\s*(\{|extends|\s+implements)')
        matches = class_pattern.findall(content)
        NC += len(matches)

        # calculate the ANA metric
        attribute_pattern = re.compile(
            r'([A-Za-z0-9_]+)\s+[A-Za-z0-9_]+\s*=\s*')
        matches = attribute_pattern.findall(content)
        ANA += len(matches)

        # calculate the ANM, ANSM, ANGM, and ANCM metrics
        method_pattern = re.compile(r'([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\(')
        methods = method_pattern.findall(content)
        for method in methods:
            ANM += 1
            method_name = method[1]
            if method_name.startswith('set'):
                ANSM += 1
            elif method_name.startswith('get'):
                ANGM += 1
            else:
                ANCM += 1

        # calculate the NGen and NAssoc metrics
        for match in matches:
            class_name = match[0]
            if re.match('[A-Z]', class_name[0]):
                NGen += 1
            else:
                NAssoc += 1

    # calculate the average metrics
    if NC > 0:
        ANA /= NC
        ANM /= NC
        ANSM /= NC
        ANGM /= NC
        ANCM /= NC

    return [repo_name, url, "{:.4f}".format(LOC), "{:.4f}".format(NC), "{:.4f}".format(ANA), "{:.4f}".format(ANM), "{:.4f}".format(ANSM), "{:.4f}".format(ANGM), "{:.4f}".format(ANCM), "{:.4f}".format(NGen), "{:.4f}".format(NAssoc)]


if __name__ == '__main__':
    with open('rep50S.txt', 'r') as input_file, open('results2.csv', 'w', newline='') as output_file:
        # create the repositories directory if it does not exist
        if not os.path.exists('repositories'):
            os.makedirs('repositories')

        # Initialize a CSV writer
        writer = csv.writer(output_file)

        # Write the header row
        writer.writerow(['Repository Name', 'Repository URL', 'LOC', 'NC', 'ANA', 'ANM',
                        'ANSM', 'ANGM', 'ANCM', 'NGen', 'NAssoc'])

        # Loop over each repository URL in the input file
        for url in input_file:
            # Calculate the metrics for the repository
            metrics = calculate_metrics(url.replace('\n', ''))

            # Write the metrics to the output file
            writer.writerow(metrics)

        # Delete the repositories directory
        # shutil.rmtree('repositories')
