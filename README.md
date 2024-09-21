# Data Metrics

This is a simple Python script that calculates data metrics across GitHub repositories using links in urls.txt. The following metrics are calculated:

LOC, NOC, NCL, RFC, CBO, DIT, LCOM5, WMC

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/shebnik/data_metrics.git
   ```
2. Create a `urls.txt` file in the root of the repository.
3. Run the script:
   ```
   python3 metric_calc.py
   ```
4. The script will clone each repository specified in the `urls.txt` file into the `/repositories` folder and write the metrics results into a `results.csv` file.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
