import os

search_service_name = os.environ["SEARCH_SERVICE_NAME"]
search_endpoint = f"https://{search_service_name}.search.windows.net"

print(search_endpoint)