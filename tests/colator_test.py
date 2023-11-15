import unittest
from unittest.mock import MagicMock

from checker import process_input


class TestProcessInput(unittest.TestCase):
    def test_process_input_returns_list(self):
        # Arrange
        input_code_string_batch = ["print('hello, world!')", "x = 5"]

        # Act
        result = process_input(input_code_string_batch)

        # Assert
        self.assertIsInstance(result, list)

    def test_process_input_returns_expected_length(self):
        # Arrange
        input_code_string_batch = ["print('hello, world!')", "x = 5"]

        # Act
        result = process_input(input_code_string_batch)

        # Assert
        self.assertEqual(len(result), len(input_code_string_batch))

    def test_process_input_calls_prepare_input(self):
        # Arrange
        input_code_string_batch = ["print('hello, world!')", "x = 5"]
        prepare_input_mock = MagicMock()

        # Act
        with unittest.mock.patch("colator.prepare_input", prepare_input_mock):
            process_input(input_code_string_batch)

        # Assert
        prepare_input_mock.assert_called_with(input_code_string_batch[0])
        self.assertEqual(prepare_input_mock.call_count, len(input_code_string_batch))

    def test_process_input_calls_prepare_inputs_for_infill(self):
        # Arrange
        input_code_string_batch = ["print('hello, world!')", "x = 5"]
        prepare_inputs_for_infill_mock = MagicMock()
        prepare_input_mock = MagicMock(return_value={})

        # Act
        with unittest.mock.patch(
            "colator.prepare_inputs_for_infill", prepare_inputs_for_infill_mock
        ):
            with unittest.mock.patch("colator.prepare_input", prepare_input_mock):
                process_input(input_code_string_batch)

        # Assert
        prepare_inputs_for_infill_mock.assert_called_with(input_code_string_batch[0], {})
        self.assertEqual(
            prepare_inputs_for_infill_mock.call_count, len(input_code_string_batch)
        )
