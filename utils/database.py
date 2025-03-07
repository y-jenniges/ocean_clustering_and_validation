import pandas as pd


def get_table_as_df(conn, table_name, columns=None):
    """ Get a table from the database as pandas dataframe.
    :param table_name (str): Name of the table to fetch.
    :param conn (sqlite3.Connection): Connection to the database that stores the table.
    :param columns (list<str>): List of columns to query. If None, all columns will be queried. Default is None.
    :return df (pandas.DataFrame): Dataframe containing the table information.
    """
    col_selection = "*"
    if columns:
        col_selection = ", ".join(columns)

    query = f"select {col_selection} from {table_name};"
    cur = conn.cursor()
    ex = cur.execute(query)
    cols = [description[0] for description in cur.description]

    result = ex.fetchall()
    df = pd.DataFrame(result, columns=cols)
    return df


def get_num_samples(conn, table_names):
    """ Get the number of samples contained in the given table.
    :param conn (sqlite3.Connection): Connection to the database.
    :param table_names (list<str>): List of table names to count.
    :return: num_samples (int): Summed number of sample in the given tables.
    """
    cur = conn.cursor()

    query = "SELECT " + " + ".join([f"(SELECT COUNT(*) FROM {x})" for x in table_names]) + ";"
    ex = cur.execute(query)
    num_samples = ex.fetchall()[0][0]

    cur.close()

    return num_samples
