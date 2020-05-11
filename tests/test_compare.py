"""
Basic tests for Titanic comparator module.
"""
import json
import os
import unittest
from unittest import mock

from titanic_comparator import Compare, ComparisionException


test_dir = os.path.dirname(os.path.realpath(__file__))


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data_path, status_code):
            self.json_data = None
            if json_data_path:
                with open(rf"{test_dir}\{json_data_path}", 'r') as f:
                    raw_data = json.load(f)

                self.json_data = {
                    "records": raw_data
                }
            self.status_code = status_code

        def json(self):
            return self.json_data

    if args[0] == 'http://test.com/small':
        return MockResponse(r'test_data\small.json', 200)
    elif args[0] == 'http://test.com/additional_column':
        return MockResponse(r'test_data\additional_column.json', 200)
    elif args[0] == 'http://test.com/missing_column':
        return MockResponse(r'test_data\missing_column.json', 200)
    elif args[0] == 'http://test.com/small_reordered':
        return MockResponse(r'test_data\small_reordered.json', 200)
    elif args[0] == 'http://test.com/small_with_differences':
        return MockResponse(r'test_data\small_with_differences.json', 200)

    return MockResponse(None, 404)


class CompareTest(unittest.TestCase):

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_wrong_response(self, mock_get):
        """
        Tests:
        - that wrong requests (with status code != 200) raises custom exception
        """
        # print(test_dir)
        with self.assertRaises(ComparisionException):
            comp = Compare(
                expected_data_path=rf'{test_dir}\test_data\small.json',
                actual_data_url='http://test.com/non_existing_resource'
            )

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_reading_from_json(self, mock_get):
        """
        Tests:
        - reading from json file (expected data)
        - IN operator (actual data == expected data)
        - contains_same_data method (actual data == expected data)
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.json',
            actual_data_url='http://test.com/small'
        )

        self.assertTrue(comp.contains_same_data())
        self.assertTrue(comp.actual_data_in_expected_data())

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_reading_from_csv(self, mock_get):
        """
        Tests:
        - reading from csv file (expected data)
        - IN operator - data in small.json (mocked response) is inside small.csv (expected)
        - contains_same_data method (actual data != expected data)
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/small'
        )

        self.assertTrue(comp.actual_data_in_expected_data())
        self.assertFalse(comp.contains_same_data())

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_additional_column_in_actual_data(self, mock_get):
        """
        Tests:
        - that additional column in response is detected
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/additional_column'
        )

        self.assertFalse(comp.actual_data_in_expected_data())
        self.assertFalse(comp.contains_same_data())

        expected_additional_column = {"additional_column"}
        stats = comp.get_statistics()

        self.assertEqual(stats["additional_columns"], expected_additional_column)
        self.assertEqual(stats["missing_columns"], set())

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_missing_column_in_actual_data(self, mock_get):
        """
        Tests:
        - that missing column in response is detected
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/missing_column'
        )

        self.assertFalse(comp.actual_data_in_expected_data())
        self.assertFalse(comp.contains_same_data())

        expected_missing_column = {"sibsp"}
        stats = comp.get_statistics()

        self.assertEqual(stats["missing_columns"], expected_missing_column)
        self.assertEqual(stats["additional_columns"], set())

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_one_to_one_cmp_no_differences(self, mock_get):
        """
        Tests:
        - that rows order in actual data is not important when using one-to-one comparision
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/small_reordered'
        )
        self.assertTrue(comp.actual_data_in_expected_data())
        self.assertTrue(comp.contains_same_data())

        stats = comp.get_statistics()

        self.assertEqual(len(stats["common_rows"]), 7)
        self.assertEqual(len(stats["additional_rows"]), 0)
        self.assertEqual(len(stats["missing_rows"]), 0)

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_one_to_one_cmp_with_differences(self, mock_get):
        """
        Tests:
        - that rows order in actual data is not important when using one-to-one comparision

        Summary of differences in actual data:
        for name="Collander, Mr. Erik Gustaf", pclass 2->1, passengerid 343 -> 342
        for name="Jensen, Mr. Hans Peder", age 20.0 -> 60.0, survived No -> Yes
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/small_with_differences'
        )
        self.assertFalse(comp.actual_data_in_expected_data())
        self.assertFalse(comp.contains_same_data())

        stats = comp.get_statistics()

        self.assertEqual(len(stats["common_rows"]), 5)
        self.assertEqual(len(stats["additional_rows"]), 2)
        self.assertEqual(len(stats["missing_rows"]), 2)

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_on_column_cmp_no_differences(self, mock_get):
        """
        Tests:
        - that rows order in actual data is not important when using on-column comparision
        - comparing on 1 column works as expected
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/small_reordered',
            columns="Name"
        )
        self.assertTrue(comp.actual_data_in_expected_data())
        self.assertTrue(comp.contains_same_data())

        stats = comp.get_statistics()

        self.assertEqual(len(stats["common_rows"]), 7)
        self.assertEqual(len(stats["additional_rows"]), 0)
        self.assertEqual(len(stats["missing_rows"]), 0)
        self.assertEqual(len(stats["rows_with_differences"]), 0)
        self.assertEqual(len(stats["differences_in_rows_with_common_key"]), 0)

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_on_column_cmp_with_differences(self, mock_get):
        """
        Tests:
        - that rows order in actual data is not important when using on-column comparision
        - comparing on 2 columns work as expected

        Summary of differences in actual data:
        for name="Collander, Mr. Erik Gustaf", pclass 2->1, passengerid 343 -> 342
        for name="Jensen, Mr. Hans Peder", age 20.0 -> 60.0, survived No -> Yes
        """
        comp = Compare(
            expected_data_path=rf'{test_dir}\test_data\small.csv',
            actual_data_url='http://test.com/small_with_differences',
            columns=["Name", "PassengerId"]
        )
        self.assertFalse(comp.actual_data_in_expected_data())
        self.assertFalse(comp.contains_same_data())

        stats = comp.get_statistics()

        self.assertEqual(len(stats["common_rows"]), 5)
        self.assertEqual(len(stats["additional_rows"]), 1)  # key ("Collander, Mr. Erik Gustaf", 342)
        self.assertEqual(len(stats["missing_rows"]), 1)  # key ("Collander, Mr. Erik Gustaf", 343)
        self.assertEqual(len(stats["rows_with_differences"]), 1)  # common key:  ("Jensen, Mr. Hans Peder", 641)
        self.assertEqual(len(stats["differences_in_rows_with_common_key"]), 2)  # age and survived


if __name__ == '__main__':
    unittest.main()
