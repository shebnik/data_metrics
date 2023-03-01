# Data Metrics

This is a simple Python script that calculates data metrics across GitHub repositories using links in urls.txt. The following metrics are calculated:

- LOC is the number of lines of program code.
- NC is the total number of classes.
- ANA is the average number of attributes.
- ANM is the average number of methods.
- ANSM is the average number of set methods.
- ANGM is the average number of get methods.
- ANCM is the average number of design methods.
- NGen and NAssoc are types of relations between classes (generalization / association).

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/shebnik/data_metrics.git
   ```
2. Create a `urls.txt` file in the root of the repository.
3. You can change the language in the script. For example:
    ```python
    language = "java"
    ```

    or

    ```python
    language = "py"
    ```
4. Run the script:
   ```
   python3 data_metrics.py
   ```
5. The script will clone each repository specified in the `urls.txt` file into the `/repositories` folder and write the metrics results into a `results.csv` file.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
