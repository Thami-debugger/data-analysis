import unittest
import pandas as pd
from utils import load_data
import io


class TestUtils(unittest.TestCase):
    def test_load_data_csv(self):
        # Create a dummy CSV file
        csv_data = "col1,col2\n1,2\n3,4"
        csv_file = io.StringIO(csv_data)
        csv_file.name = "test.csv"

        # Load the data
        df = load_data(csv_file)

        # Check that the data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.columns.tolist(), ["col1", "col2"])
        self.assertEqual(df["col1"].tolist(), [1, 3])
        self.assertEqual(df["col2"].tolist(), [2, 4])

    def test_load_data_excel(self):
        # Create a dummy Excel file
        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
            pd.DataFrame({"col1": [1, 3], "col2": [2, 4]}).to_excel(writer, index=False)
        excel_data.seek(0)
        excel_file = excel_data
        excel_file.name = "test.xlsx"


        # Load the data
        df = load_data(excel_file)

        # Check that the data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.columns.tolist(), ["col1", "col2"])
        self.assertEqual(df["col1"].tolist(), [1, 3])
        self.assertEqual(df["col2"].tolist(), [2, 4])

if __name__ == "__main__":
    unittest.main()
