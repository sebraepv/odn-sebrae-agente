# import pyodbc
# from azure.identity import DefaultAzureCredential


# conn_str = (
#     "Driver={ODBC Driver 18 for SQL Server};"
#     "Server=tcp:sc-sandbox-sqlsrv.database.windows.net,1433;"
#     "Database=odn-database;"
#     "Authentication=ActiveDirectoryInteractive;"
#     "Encrypt=yes;"
#     "TrustServerCertificate=no;""
# )

# print(conn_str)

# conn = pyodbc.connect(conn_str)
# # print("OK")


import logging

logging.basicConfig(level=logging.DEBUG)

from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

token = credential.get_token(
    "https://management.azure.com/.default"
)

print("OK")