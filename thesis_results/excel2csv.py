import pandas as pd


def xlsx_to_csv_pd():
    data_xls = pd.read_excel('results.xlsx', index_col=0)
    data_xls.to_csv('exp_results.csv', encoding='utf-8')


if __name__ == '__main__':
    xlsx_to_csv_pd()
