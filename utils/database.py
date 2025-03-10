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


def does_table_exist(conn, table_name, table_type="table"):
    """
    Checks if a table/view with the given name exists. Warning: Make sure that the casing is correct. Function is
    case-sensitive!

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Check if this table name is already in the database.
        table_type (str): Type of the database structure to check, e.g. 'view'. Default is 'table'.
    Returns:
        does_table_exist (bool): If the table/view exists in the database.
    """
    cur = conn.cursor()
    query = f"select name from sqlite_master where type='{table_type}' AND name='{table_name}';"
    result = cur.execute(query).fetchall()

    return True if result else False


def remove_tables_like(conn, like_pattern="E|_%", escape_char="|", table_type="view", tables_except=None):
    """
    Removes views/tables whose name match the given like_pattern.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        like_pattern (str): Table names should conform to this string pattern. If None, all parameter tables are
        queried. Default is 'E|_%'
        escape_char (str): Escape char used for the like pattern. If None, sqlite3 default is used. Default is '|'.
        table_type (str): Type of the structure (view or table) to remove. Default is 'view'.
        tables_except (list<str>): Remove all tables matching the like pattern except the ones specified in this list.
        Default is None.
    """
    # get views/tables to remove
    table_names = get_names_of_all_parameter_tables(conn, like_pattern, escape_char, table_type=table_type,
                                                    include_digits=True)
    cur = conn.cursor()

    if tables_except:
        for te in tables_except:
            if te in table_names:
                table_names.remove(te)

    for vtn in table_names:
        query = f"drop {table_type} {vtn};"
        print(f"Structure.remove_tables_like: {query}")
        cur.execute(query)


def get_names_of_all_parameter_tables(conn, like_pattern="P|_%", escape_char="|", include_digits=False,
                                      table_type="table"):
    """
    Get all tables conforming with the given like_pattern.

    Args:
        conn (sqlite3.Connection): Connection to the sqlite3 database.
        like_pattern (str): Table names should conform to this string pattern. If None, all table names are queried.
        Default is 'P|_%'
        escape_char (str): Escape char used for the like pattern. If None, sqlite3 default are used. Default is '|'.
        include_digits (bool): Weather tables names including numbers should be returned as well.
        table_type (str): The table type to look for (e.g. 'view'). Default is 'table'.
    Returns:
        param_table_names (list<str>): A list of table names that conform to the given pattern.
    """
    digits_filter = ""
    if not include_digits:
        digits_filter = "and name not glob '*_[0-9]*'"

    if like_pattern and escape_char:
        query = f"select name from sqlite_master " \
                f"where type='{table_type}' and name like '{like_pattern}' escape '{escape_char}' " \
                f"{digits_filter};"
    elif like_pattern:
        query = f"select name from sqlite_master " \
                f"where type='{table_type}' and name like '{like_pattern}' " \
                f"{digits_filter};"
    else:
        query = f"select name from sqlite_master " \
                f"where type='{table_type}' " \
                f"{digits_filter};"

    print(f"Information: {query}")
    ex = conn.cursor().execute(query)
    result = ex.fetchall()
    param_table_names = [entry[0] for entry in result]

    return param_table_names
