import os.path
import sqlite3
import logging
import gsw
from utils.database import get_table_as_df, get_num_samples
from preparation.units import UnitsConverter


def grid_data():
    return


def impute_data():
    return


def copy_and_filter_tables(tables, quality_flags, dest_cursor, source_db_path):
    """ Copy tables from a database to another database and filter for quality.
    :param tables: Table names to copy.
    :param quality_flags: Specification for PQF1, PQF2 and SQF used to filter the data for quality.
    :param dest_cursor: Cursor of the destination database.
    :param source_db_path: Cursor of the source database.
    """
    logging.info("Copy and filter tables...")

    # Attach the source database to the destination database
    dest_cursor.execute(f"ATTACH DATABASE '{source_db_path}' AS source_db")

    # Copy tables to destination database (ensure that temperature and salinity are always copied)
    logging.info("Copying tables to new database...")
    for parameter in list(set(tables + ["P_TEMPERATURE", "P_SALINITY"])):
        query = f"CREATE TABLE IF NOT EXISTS {parameter} AS "\
                f"SELECT * FROM source_db.{parameter} "\
                f"WHERE {' and '.join([x[0] + x[1] for x in quality_flags])};"
        logging.info("  " + query)
        print(query)
        dest_cursor.execute(query)

    # Copy STATION, CRUISE, DATABASE_TABLES for location and time information and info on tables
    for table in ["STATION", "CRUISE", "DATABASE_TABLES"]:
        query = f"CREATE TABLE IF NOT EXISTS {table} AS " \
                f"SELECT * FROM source_db.{table};"
        logging.info("  " + query)
        print(query)
        dest_cursor.execute(query)


def add_location_time(tables, cursor):
    """ Add latitude, longitude and dateandtime information to tables.
    :param tables: Names of tables to add the location and time information to.
    :param cursor: Cursor of the database.
    """
    logging.info("Adding location and time...")

    for table in tables:
        print(table)
        logging.info("  " + table)
        query = f"CREATE TABLE IF NOT EXISTS {table}_llt AS " \
                f"SELECT p.*, STATION.LATITUDE, STATION.LONGITUDE, STATION.DATEANDTIME " \
                f"FROM {table} AS p " \
                f"LEFT JOIN STATION ON p.ID=STATION.ID; "
        logging.info("  " + query)
        cursor.execute(query)

        # Drop initial tables
        query = f"DROP TABLE {table};"
        cursor.execute(query)

        # Rename new tables
        query = f"ALTER TABLE {table}_llt RENAME TO {table};"
        cursor.execute(query)

    # Clean database
    logging.info("  Vacuum database...")
    query = "VACUUM;"
    cursor.execute(query)


def average_over_location_time(tables, cursor):
    """ Average VAL column of a table over LATITUDE, LONGITUDE and DATEANDTIME columns.
    :param tables: Tables to compute the average for.
    :param cursor: Cursor to the database.
    """
    logging.info("Averaging over location and time...")
    for table in tables:
        logging.info("  " + table)

        # Get column names
        query = f"PRAGMA table_info({table});"
        ex = cursor.execute(query)
        cols = [x[1] for x in ex.fetchall() if x[1] != "VAL"]

        # Grouping
        query = f"CREATE TABLE {table}_avg AS " \
                f"SELECT AVG(VAL) AS VAL, {', '.join(cols)} " \
                f"FROM {table} GROUP BY LATITUDE, LONGITUDE, LEV_M, DATEANDTIME;"
        logging.info("  " + query)
        cursor.execute(query)

        # Drop initial tables
        query = f"DROP TABLE {table};"
        cursor.execute(query)

        # Rename new tables
        query = f"ALTER TABLE {table}_avg RENAME TO {table};"
        cursor.execute(query)

    # Clean database
    logging.info("  Vacuum database...")
    query = "VACUUM;"
    cursor.execute(query)


def add_temperature_salinity(tables, cursor):
    """ Joining temperature and salinity information to given tables.
    :param tables: Tables to add temperature and salinity to.
    :param cursor: Cursor to the database.
    """
    logging.info("Averaging over location and time...")
    for table in tables:
        print(table)
        logging.info("  " + table)

        # Adding temperature and salinity information as required
        if table == "P_TEMPERATURE":
            ts_select_statement = ", s.VAL as salinity"
            ts_join_statement = f"LEFT JOIN P_SALINITY s ON (p.LEV_M=s.LEV_M and p.ID=s.ID)"
        elif table == "P_SALINITY":
            ts_select_statement = ", t.VAL as temperature"
            ts_join_statement = f"LEFT JOIN P_TEMPERATURE t ON (p.LEV_M=t.LEV_M and p.ID=t.ID)"
        else:
            ts_select_statement = ", t.VAL as temperature, s.VAL as salinity"
            ts_join_statement = f"LEFT JOIN P_TEMPERATURE t ON (p.LEV_M=t.LEV_M and p.ID=t.ID) " \
                                f"LEFT JOIN P_SALINITY s ON (p.LEV_M=s.LEV_M and p.ID=s.ID)"

        # Add temperature and salinity information
        query = f"CREATE TABLE {table}_ts AS " \
                f"SELECT p.* {ts_select_statement} " \
                f"FROM {table} p " \
                f"{ts_join_statement};"
        logging.info("  " + query)
        cursor.execute(query)

        # Drop initial tables
        query = f"DROP TABLE {table};"
        cursor.execute(query)

        # Rename new tables
        query = f"ALTER TABLE {table}_ts RENAME TO {table};"
        cursor.execute(query)

    # Clean database
    logging.info("  Vacuum database...")
    query = "VACUUM;"
    cursor.execute(query)


def convert_units(tables, connection):
    logging.info("Converting units...")

    # Define converter and convert
    df_default_units = get_table_as_df(connection, "DATABASE_TABLES")
    df_units = get_table_as_df(connection, "UNITS").sort_values("ID")
    units_converter = UnitsConverter(connection, df_default_units, df_units, value_column="VAL")
    units_converter.convert_units(tables, use_density=True, override_old_tables=True)

    # Tidy up disc usage
    logging.info("    Vacuum database...")
    query = "VACUUM;"
    cursor = connection.cursor()
    cursor.execute(query)
    cursor.close()


def t_to_pot_t(connection):
    """ Convert temperature to potential temperature.
    :param connection: (sqlite3.Connection) Connection to database.
    """
    logging.info("Convert temperature to potential temperature...")
    t_table = "P_TEMPERATURE"
    cursor = connection.cursor()

    # Get temperature table
    df_t = get_table_as_df(connection, t_table,
                           columns=["ID", "LATITUDE", "LONGITUDE", "LEV_M", "LEV_DBAR", "VAL", "salinity"])

    # Apply conversion and drop values that could not be converted and lead to NaN entries
    df_t["salinity_absolute"] = gsw.SA_from_SP(SP=df_t["salinity"], p=df_t["LEV_DBAR"] - 10.1325,
                                               lon=df_t["LONGITUDE"], lat=df_t["LATITUDE"])
    df_t["pot_temperature"] = gsw.conversions.pt0_from_t(SA=df_t["salinity_absolute"], t=df_t["VAL"],
                                                         p=df_t["LEV_DBAR"])
    count_old = len(df_t)
    df_t.dropna(subset=["pot_temperature"], inplace=True)
    df_t.drop("salinity_absolute", axis="columns", inplace=True)
    count_new = len(df_t)

    # Output how many values were not converted and thus dropped
    logging.info(f"  Conversion not possible for {count_old-count_new} values.")

    # Write potential temperature table to database
    df_t.to_sql("temp", connection, if_exists="fail")

    # Get columns of original temperature table
    q = f"PRAGMA table_info({t_table});"
    ex = cursor.execute(q)
    cols = ["t." + x[1] for x in ex.fetchall() if x[1] != "VAL"]

    # Combine original table and temp table (with potential temperature) so that all columns are in the table
    q = f"CREATE TABLE temp2 AS SELECT temp.pot_temperature as VAL, {', '.join(cols)} " \
        f"FROM temp " \
        f"LEFT JOIN {t_table} as t USING(ID, LEV_M); "
    cursor.execute(q)

    # Drop original table
    q = f"DROP TABLE {t_table};"
    cursor.execute(q)
    q = f"DROP TABLE temp;"
    cursor.execute(q)

    # Renaming
    q = f"ALTER TABLE temp2 RENAME TO {t_table};"
    cursor.execute(q)

    # Clear database
    logging.info("  Vacuum database...")
    q = "VACUUM;"
    cursor.execute(q)

    # Close cursor
    cursor.close()


def prepare_database(parameters, quality_flags, source_db_path="../../data/comfort.sqlite", dest_db_path="output/custom.db"):
    """ Store data of interest in new database and prepare it for further processing. Operations are executed in the
    database, except for unit conversions.
    :param parameters: Parameter tables of interest
    :param quality_flags: List of quality flags to filter for
    :param source_db_path: Path to source database
    :param dest_db_path: Path to destination database
    """

    # Only prepare the database if the new database does not exist yet
    if not os.path.exists(dest_db_path):
        # Define quality flags to filter for
        if not quality_flags:
            quality_flags = [["pqf1", ">0"], ["pqf2", ">2"], ["sqf", ">=-1"]]

        # Connect to COMFORT database
        source_conn = sqlite3.connect(source_db_path)
        source_cursor = source_conn.cursor()
        logging.info(f"Number of samples (original tables): {get_num_samples(conn=source_conn, table_names=parameters)}")

        # Connect to a new database (or create it if it doesn't exist)
        dest_conn = sqlite3.connect(dest_db_path)
        dest_cursor = dest_conn.cursor()

        # Copy desired tables to new database and filter for quality
        copy_and_filter_tables(tables=parameters, quality_flags=quality_flags,
                               dest_cursor=dest_cursor, source_db_path=source_db_path)
        logging.info(f"Number of samples (filtered tables): {get_num_samples(conn=dest_conn, table_names=parameters)}")

        # Add latitude, longitude, dateandtime information to temperature and salinity tables
        add_location_time(tables=["P_TEMPERATURE", "P_SALINITY"], cursor=dest_cursor)

        # Average temperature and salinity values at the same time and location
        average_over_location_time(tables=["P_TEMPERATURE,  P_SALINITY"], cursor=dest_cursor)
        logging.info(f"Number of samples (TS averaged over time and location): {get_num_samples(conn=dest_conn, table_names=parameters)}")

        # Add temperature and salinity information to all tables
        add_temperature_salinity(tables=parameters, cursor=dest_cursor)

        # Add latitude, longitude, dateandtime information to other tables
        add_location_time(tables=[p for p in parameters if p not in ["P_TEMPERATURE,  P_SALINITY"]], cursor=dest_cursor)

        # Unit conversions
        convert_units(tables=parameters, connection=dest_conn)

        # Average values at same location and position
        average_over_location_time(tables=[p for p in parameters if p not in ["P_TEMPERATURE",  "P_SALINITY"]], cursor=dest_cursor)
        logging.info(f"Number of samples (all averaged): {get_num_samples(conn=dest_conn, table_names=parameters)}")

        # Convert temperature to potential temperature
        t_to_pot_t(connection=dest_conn)
        logging.info(f"Number of samples (t-->pot_t): {get_num_samples(conn=dest_conn, table_names=parameters)}")
    else:
        logging.info(f"Database already exists at {dest_db_path}. Skipping database preparation.")


def load_data(parameters):
    return
