"""
Library for Titanic passenger comparision.
Implemented features:
- support for '.csv' and '.json' format for local (expected) data
- support for REST API calls for fetching actual data
- support for one-to-one comparision
- support for text report (both to console and file)
- support for comparision by given rows
- testing if actual data is inside expected data

Features not yet implemented:
- support for exporting comparison results in table like format
"""
import json
from os import path
from io import StringIO
from shutil import copyfileobj
from datetime import datetime

import pandas as pd
import numpy as np
import requests


# set display options for pandas to nicely display titanic data table
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ComparisionException(Exception):
    pass


def load_df(data_path: str) -> pd.DataFrame:
    """
    Loads data into pandas DataFrame from local file. Currently two file
    types are supported : '.csv' and '.json'
    :param data_path: path to file with data - must be in '.csv' (with ';' as separator) or '.json' format
    :return: Resulting DataFrame
    """
    _, extension = path.splitext(data_path)

    if extension == ".csv":
        df = pd.read_csv(data_path, sep=";")  # provided site uses ';' as a separator in Titanic data
        normalize_column_names(df)

        return df

    if extension == ".json":
        with open(data_path, 'r') as file:
            raw_data = file.read()

        records = [record["fields"] for record in json.loads(raw_data)]

        return pd.DataFrame(records)

    raise ComparisionException(f"Unsupported file type: {extension}")


def fetch_df(url: str) -> pd.DataFrame:
    """
    Fetches data from given url and converts it to pandas DataFrame.
    :param url: REST endpoint with data (currently supported site:
    'https://public.opendatasoft.com/explore/dataset/titanic-passengers/api/')
    :return: Resulting DataFrame
    """
    response = requests.get(url)

    if response.status_code != 200:
        raise ComparisionException(f"Response status code was {response.status_code}")

    data = response.json()

    records = [record["fields"] for record in data["records"]]
    df = pd.DataFrame(records)
    normalize_column_names(df)

    return df


def normalize_column_names(df: pd.DataFrame) -> None:
    """
    Converts column names in given DataFrame
    """
    df.columns = list(map(str.lower, df.columns))


class Compare:
    """
    Main class for Titanic data comparision.

    In general, the idea for comparing two data sets is to convert them to pandas
    DataFrame, outer merge those frame with indicator=True (it will create additional column,
    '_merge', with information if given row is common to both data sets or specific to one of
    them), and examine output of the merge command using content of '_merge' column.

    See:
    self._compare_rows_one_to_one - implementation of comparision WITHOUT rows
    self._compare_rows_using_columns - implementation of comparision WITH rows
    """
    def __init__(self, expected_data_path: str, actual_data_url: str, columns=None):
        """
        Performs comparsion of data from given file and REST API endpoint.
        :param expected_data_path: full path to file with expected data ('.csv' and '.json' files are supported)
        :param actual_data_url: url address for REST API call with expected data
        :param columns: (optional) list of column names which should be used for comparision
        """
        self._fare_multiplier = 100  # see self._pre_process_df for explanation
        self._expected_df = load_df(expected_data_path)
        self._pre_process_df(self._expected_df)
        self._expected_data_source = expected_data_path

        self._actual_df = fetch_df(actual_data_url)
        self._pre_process_df(self._actual_df)
        self._actual_data_source = actual_data_url

        # converting to columns to list if string and working with only one type
        # (list instead of list OR str) makes it later a lot easier e.g. to test for equality
        self._on_columns = [columns] if isinstance(columns, str) else columns
        if columns:
            # convert column names to lower case to be consistent with json data naming
            self._on_columns = list(map(str.lower, self._on_columns))

        self._common_columns = None
        self._columns_only_in_expected_df = None
        self._columns_only_in_actual_df = None

        self._merged_df = None
        self._records_only_in_expected_df = None
        self._records_only_in_actual_df = None
        self._records_exactly_same = None
        self._records_with_differences = None  # only when using on columns
        self._differences = None

        self._perform_comparision()

    def _pre_process_df(self, df):
        """
        Multiplies 'fare' column by constant factor and converts it to int. Doing so,
        we get rid of float comparision which are painful to handle.
        """
        if 'fare' not in df.columns:
            return

        df['fare'] = np.round(df['fare'] * self._fare_multiplier).astype(int)

    def _post_process_df(self, df, fare_column_name='fare'):
        """
        Converts 'fare' column back to original values (floats)
        """
        if fare_column_name not in df.columns:
            return

        df[fare_column_name] = df[fare_column_name] / self._fare_multiplier

    def contains_same_columns(self):
        """
        Tests if actual data contains the same columns as expected data.
        :return: True if columns in actual data == columns in expected data; False otherwise
        """
        return len(self._columns_only_in_expected_df) == 0 and len(self._columns_only_in_actual_df) == 0

    def contains_same_data(self) -> bool:
        """
        Tests if provided data sources contains data sets with EXACTLY the same data (row order is NOT important)
        :return: True if expected == actual data; False otherwise
        """
        if not self.contains_same_columns():
            return False

        if self._on_columns and len(self._differences):
            # actual data contains differences in some columns, data sets are not the same
            return False

        return len(self._records_only_in_expected_df) == 0 and len(self._records_only_in_actual_df) == 0

    def actual_data_in_expected_data(self) -> bool:
        """
        Tests if actual data is contained inside expected data (no additional rows, same columns)
        :return: True if actual data is IN expected data, False otherwise
        """
        if not self.contains_same_columns():
            return False

        if self._on_columns and len(self._differences):
            return False

        return len(self._records_only_in_actual_df) == 0

    def report(self, target_file_path=None) -> None:
        """
        Creates and outputs to console report with brief summary of comparison.
        Optionally, report can be saved to file if file is provided.
        :param target_file_path: (optional) if provided, report will be saved to this file
        """
        report = StringIO()

        def _add_text_to_report(text):
            report.write(text)
            report.write("\n")

        def _add_df_to_report(df):
            report.write("\n")
            report.write(str(df))
            report.write("\n\n")

        def _general_spacer():
            _add_text_to_report('*' * 120)

        def _sub_spacer():
            _add_text_to_report("-" * 120)

        _general_spacer()
        _add_text_to_report("Titanic data comparison report")
        _sub_spacer()

        _add_text_to_report("General summary:")
        _add_text_to_report(f"Created on: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
        if self._on_columns:
            _add_text_to_report("Comparison type: on-columns")
            _add_text_to_report(f"Columns used for comparison: {self._on_columns}")
        else:
            _add_text_to_report("Comparison type: one-to-one")

        _add_text_to_report(f"Actual data match EXACTLY expected data: {'YES' if self.contains_same_data() else 'NO'}")
        _add_text_to_report(
            f"Actual columns match EXACTLY expected columns: {'YES' if self.contains_same_columns() else 'NO'}"
        )
        _add_text_to_report(f"Expected data loaded from: {self._expected_data_source}")
        _add_text_to_report(f"Actual data fetched from: {self._actual_data_source}")
        _sub_spacer()

        _add_text_to_report("Columns comparison details:")
        _add_text_to_report(f"# of common columns for expected and actual data: {len(self._common_columns)}")
        _add_text_to_report(f"# of columns which occur ONLY in expected data: {len(self._columns_only_in_expected_df)}")
        if self._columns_only_in_expected_df:
            _add_text_to_report(f"Column names which occur ONLY in expected data: {self._columns_only_in_expected_df}")

        _add_text_to_report(f"# of additional columns in actual data: {len(self._columns_only_in_actual_df)}")
        if self._columns_only_in_actual_df:
            _add_text_to_report(f"Additional column names: {self._columns_only_in_actual_df}")
        _sub_spacer()

        _add_text_to_report("Data comparison details:")
        _add_text_to_report(f"# of common records for expected and actual data: {len(self._records_exactly_same)}")
        _add_text_to_report(
            f"# of records records which occur ONLY in expected data: {len(self._records_only_in_expected_df)}"
        )
        if len(self._records_only_in_expected_df):
            _add_text_to_report("First 5 records which occur ONLY in expected data:")
            _add_df_to_report(self._records_only_in_expected_df.head())

        _add_text_to_report(f"# of additional records in actual data: {len(self._records_only_in_actual_df)}")
        if len(self._records_only_in_actual_df):
            _add_text_to_report("First 5 additional records in actual data:")
            _add_df_to_report(self._records_only_in_actual_df.head())

        if self._on_columns:
            _add_text_to_report(
                f"# of records with same keys but differences in other columns: {len(self._records_with_differences)}"
            )
            _add_text_to_report(f"# of all differences in records with common keys: {len(self._differences)}")
            if len(self._differences):
                _add_text_to_report("First 5 differences:")
                _add_df_to_report(self._differences.head())

        _general_spacer()

        # output report to console
        print(report.getvalue())

        if target_file_path:
            with open(target_file_path, 'a') as file:
                report.seek(0)
                copyfileobj(report, file)

        report.close()

    def get_statistics(self):
        """
        Generates and returns comparation statistics.
        """
        return {
            "common_columns": self._common_columns,
            "additional_columns": self._columns_only_in_actual_df,
            "missing_columns": self._columns_only_in_expected_df,
            "common_rows": self._records_exactly_same,
            "additional_rows": self._records_only_in_actual_df,
            "missing_rows": self._records_only_in_expected_df,
            "rows_with_differences": self._records_with_differences,
            "differences_in_rows_with_common_key": self._differences
        }

    def generate_comparision_summary(self):
        pass

    def _perform_comparision(self) -> None:
        """
        Actual point of performing DataFrames comparison. First, columns are compared,
        then rows. Rows comparision can be performed on-columns (more precise) or one-to-one
        (entire rows from actual data source are compared to rows from expected data).
        """
        self._compare_columns()
        if self._on_columns:
            self._compare_rows_using_columns()
        else:
            self._compare_rows_one_to_one()

    def _compare_columns(self):
        """
        Performs actual and expected data columns comparision.
        """
        expected_columns = set(self._expected_df.columns)
        actual_columns = set(self._actual_df.columns)

        self._common_columns = expected_columns & actual_columns
        self._columns_only_in_expected_df = expected_columns - actual_columns
        self._columns_only_in_actual_df = actual_columns - expected_columns

    def _compare_rows_one_to_one(self):
        """
        Performs one-to-one comparision - this means that each row (with entire content) in actual
        data will be compared to rows in expected data.
        Method populates 3 sets of data:
        self._records_exactly_same - rows occurring in expected AND actual data (without differences)
        self._records_only_in_expected_df - rows occurring ONLY in expected data (not in actual)
        self._records_only_in_actual_df - rows occurring ONLY in actual data (not in expected)

        In this mode, detailed detection of row differences is not performed.
        """
        self._merged_df = pd.merge(
            self._expected_df,
            self._actual_df,
            how="outer",
            indicator=True
        )

        self._post_process_df(self._merged_df)

        columns = list(self._common_columns)
        self._records_only_in_expected_df = self._merged_df[self._merged_df["_merge"] == "left_only"][columns]
        self._records_only_in_actual_df = self._merged_df[self._merged_df["_merge"] == "right_only"][columns]
        self._records_exactly_same = self._merged_df[self._merged_df["_merge"] == "both"][columns]

    def _compare_rows_using_columns(self):
        """
        Performs on-columns comparision, where values from given columns are treated as keys. This mode allows
        to perform more detailed comparision and detect each difference.

        Method populates following sets of data:
        self._records_exactly_same - rows occurring in expected AND actual data (without differences)
        self._records_only_in_expected_df - rows with keys which occur ONLY in expected data (not in actual)
        self._records_only_in_actual_df - rows with keys which occur ONLY in actual data (not in expected)
        self._records_with_differences - rows with keys which occur in expected AND actual data but with differences
            in other columns
        self._differences - all differences from self._records_with_differences (each difference is separate row)
        """
        def _get_columns_to_select(suffix):
            _to_select = list()
            _to_select.extend(self._on_columns)

            _other_columns = [_column for _column in self._merged_df.columns if _column.endswith(suffix)]
            _to_select.extend(_other_columns)

            return _to_select

        def _remove_suffix_from_columns(columns, suffix):
            _new_columns = list()
            for _column in columns:
                if _column.endswith(suffix):
                    _new_column = _column[:-len(suffix)]
                    _new_columns.append(_new_column)
                else:
                    _new_columns.append(_column)

            return _new_columns

        self._merged_df = pd.merge(
            self._expected_df,
            self._actual_df,
            suffixes=("_expected", "_actual"),
            on=self._on_columns,
            how="outer",
            indicator=True
        )

        columns_to_select = _get_columns_to_select("_expected")
        self._records_only_in_expected_df = self._merged_df[self._merged_df["_merge"] == "left_only"][columns_to_select]
        self._records_only_in_expected_df.columns = _remove_suffix_from_columns(
            self._records_only_in_expected_df.columns,
            "_expected"
        )

        columns_to_select = _get_columns_to_select("_actual")
        self._records_only_in_actual_df = self._merged_df[self._merged_df["_merge"] == "right_only"][columns_to_select]
        self._records_only_in_actual_df.columns = _remove_suffix_from_columns(
            self._records_only_in_actual_df.columns,
            "_actual"
        )

        # find differences in columns
        records_with_same_keys = self._merged_df[self._merged_df["_merge"] == "both"]
        self._compare_rows_for_differences(records_with_same_keys)

    def _compare_rows_for_differences(self, records_in_both_df: pd.DataFrame):
        """
        Performs value by value comparison for records with keys occurring in both expected
        and actual data.
        As a result, self._records_exactly_same, self._records_with_differences and
        self._differences are populated.
        """
        indexed_expected_df = self._expected_df.set_index(self._on_columns)
        indexed_actual_df = self._actual_df.set_index(self._on_columns)

        identical_records = list()
        records_with_differences = list()
        differences = list()

        for _, row in records_in_both_df.iterrows():
            key = tuple(row[self._on_columns].values)

            expected_record = indexed_expected_df.loc[key]
            actual_record = indexed_actual_df.loc[key]
            records_identical = True

            for column in self._common_columns:
                if column in self._on_columns:
                    continue

                expected_value = expected_record[column]
                actual_value = actual_record[column]

                if pd.isnull(expected_value) and pd.isnull(actual_value):
                    continue

                if expected_value == actual_value:
                    continue

                if column == 'fare':
                    # special case to handle 'fare' column (float type in original)
                    # convert it back to float
                    expected_value /= self._fare_multiplier
                    actual_value /= self._fare_multiplier

                # if this point is reached, values are not equal
                # they will be stored for later use
                records_identical = False
                diff = (*key, column, expected_value, actual_value)
                differences.append(diff)

            if records_identical:
                identical_records.append(row)
            else:
                records_with_differences.append(row)

        identical_df = pd.DataFrame(identical_records)
        if identical_records:
            self._records_exactly_same = identical_df.reindex(sorted(identical_df.columns), axis=1)
            self._records_exactly_same.drop("_merge", axis=1, inplace=True)
            self._post_process_df(self._records_exactly_same, fare_column_name='fare_expected')
            self._post_process_df(self._records_exactly_same, fare_column_name='fare_actual')
        else:
            # empty data frame
            self._records_exactly_same = identical_df

        df_with_differences = pd.DataFrame(records_with_differences)
        if records_with_differences:
            self._records_with_differences = df_with_differences.reindex(sorted(df_with_differences.columns), axis=1)
            self._records_with_differences.drop("_merge", axis=1, inplace=True)
            self._post_process_df(self._records_with_differences, fare_column_name='fare_expected')
            self._post_process_df(self._records_with_differences, fare_column_name='fare_actual')
        else:
            # empty df
            self._records_with_differences = df_with_differences

        self._differences = pd.DataFrame(
            differences,
            columns=[*self._on_columns, "Column with difference", "Expected", "Actual"]
        )


if __name__ == '__main__':
    project_root = r"E:\MojePliki\Programy\luxoft-titanic"

    comp = Compare(
        expected_data_path=fr"{project_root}\data\titanic-passengers.csv",
        actual_data_url="https://public.opendatasoft.com/api/records/1.0/search/?dataset=titanic-passengers&q=&rows=900&facet=survived&facet=pclass&facet=sex&facet=age&facet=embarked",
        columns=["Name", "Survived"],
    )

    comp.report(target_file_path="report.txt")

