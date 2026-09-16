import pyodbc

conn_str = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=tcp:sc-sandbox-sqlsrv.database.windows.net,1433;"
    "Database=odn-database;"
    "Authentication=ActiveDirectoryInteractive;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)

print(conn_str)

conn = pyodbc.connect(conn_str)
print("OK")

