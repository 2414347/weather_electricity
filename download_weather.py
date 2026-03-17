import os
import requests

# ==============================
# CONFIGURATION
# ==============================

TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI4ZjhmaUpyaUtDY3hmaHhzdU5vazVEekdJdFZ4amhhTWNJa05ZX2U4MnhJIn0.eyJleHAiOjE3NzE2NDMyNTYsImlhdCI6MTc3MTM4NDA1NiwianRpIjoiOTc0NjRiOGYtOGY4Mi00MGFhLWFmNDktNDM0NzkxNjhlN2VlIiwiaXNzIjoiaHR0cHM6Ly9hY2NvdW50cy5jZWRhLmFjLnVrL3JlYWxtcy9jZWRhIiwic3ViIjoiMGE2ZjVlOTAtMGM3Ni00OTVmLTgyMDQtM2U1MjQxZDYzMjg4IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoic2VydmljZXMtcG9ydGFsLWNlZGEtYWMtdWsiLCJzZXNzaW9uX3N0YXRlIjoiNGU5NzY2NGQtYjQ3NS00ZTU0LWE2YTEtMDVkOGViZThhNzZlIiwiYWNyIjoiMSIsInNjb3BlIjoiZW1haWwgb3BlbmlkIHByb2ZpbGUgZ3JvdXBfbWVtYmVyc2hpcCIsInNpZCI6IjRlOTc2NjRkLWI0NzUtNGU1NC1hNmExLTA1ZDhlYmU4YTc2ZSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiQWJkdWxsYWgga2hhbiIsInByZWZlcnJlZF91c2VybmFtZSI6ImFiZHVsbGFoa24wMDEiLCJnaXZlbl9uYW1lIjoiQWJkdWxsYWgiLCJmYW1pbHlfbmFtZSI6ImtoYW4iLCJlbWFpbCI6ImFiZHVsbGFoa24wMDFAZ21haWwuY29tIn0.Uhb8tOt7KR6yexd35NqeJ5KQeiygH9GHFzjdf7vTD9n855SEnwfnu1SDRqotUyG4a3o3bSHDZf97QGrrCOyXeKy0SGcJ_t2a_pv_Ohunf6C_ZAC2PvQ4RErDcOl-X5nKJ48ug7Xbye6Q0h-7bFDAgTmuaK7Rusp4AWov-swm66MhiE9K6trBYE6BUEZNuHvYUcS7pL0ch0QTbKbuN8Te0cDqiEyhrOebeb4dmR6YcGkxSELa6T8eTWAR6nZ_9lodZQKogCtGoRCZT0z_0Fdp8RCq01EnfIAdiEL6fWH6KLjHT3_W9uEyjyVMU7-Wm9eE_A4hqOAeSF6ejlcpBqZDUw"

BASE_URL = "https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-weather-obs/dataset-version-202507"

START_YEAR = 2009
END_YEAR = 2024

OUTPUT_DIR = "weather_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}"
}

# ==============================
# STATION LIST (50 Stations)
# Format:
# (county, src_id, station_file_name)
# ==============================

STATIONS = [

    # --- Core South / South East ---
    ("east-sussex", "00808", "eastbourne"),
    ("west-sussex", "00782", "bognor-regis"),
    ("surrey", "00719", "wisley"),
    ("greater-london", "00708", "heathrow"),
    ("greater-london", "00721", "kew"),
    ("kent", "00744", "east-malling"),
    ("berkshire", "00641", "reading"),
    ("oxfordshire", "00625", "oxford"),
    ("hampshire", "00718", "southampton"),
    ("sussex", "00733", "brighton"),

    # --- South West ---
    ("devon", "01371", "teignmouth"),
    ("devon", "00750", "exeter"),
    ("devon", "00764", "plymouth"),
    ("cornwall", "01418", "bude"),
    ("wiltshire", "00888", "larkhill"),

    # --- Midlands ---
    ("west-midlands", "00421", "birmingham"),
    ("warwickshire", "00432", "coventry"),
    ("staffordshire", "00495", "stoke-on-trent"),
    ("nottinghamshire", "00475", "nottingham"),
    ("leicestershire", "00483", "leicester"),
    ("derbyshire", "00539", "buxton"),
    ("lincolnshire", "00567", "lincoln"),
    ("bedfordshire", "00458", "woburn"),
    ("hertfordshire", "00471", "rothamsted"),

    # --- North England ---
    ("durham", "00326", "durham"),
    ("northumberland", "00310", "morpeth-cockle-park"),
    ("west-yorkshire", "00340", "leeds"),
    ("south-yorkshire", "00525", "sheffield"),
    ("north-yorkshire", "00390", "york"),
    ("merseyside", "00357", "liverpool"),
    ("greater-manchester", "00279", "manchester"),
    ("lancashire", "00286", "blackpool"),

    # --- Wales ---
    ("glamorganshire", "00801", "cardiff"),
    ("west-glamorgan", "00814", "swansea"),
    ("clwyd", "00828", "wrexham"),
    ("monmouthshire", "00844", "abergavenny"),
    ("gwynedd", "00859", "bangor"),
    ("pembrokeshire", "01222", "tenby"),

    # --- Scotland ---
    ("dumfriesshire", "01023", "eskdalemuir"),
    ("fife", "00235", "leuchars"),
    ("shetland", "00009", "lerwick"),
    ("lanarkshire", "00212", "glasgow"),
    ("lothian", "00214", "edinburgh"),
    ("aberdeenshire", "00112", "aberdeen"),

    # --- Agriculture / Research Sites ---
    ("cambridgeshire", "00454", "cambridge-botanic-garden"),
    ("hereford-and-worcester", "00671", "ross-on-wye"),
    ("leicestershire", "00554", "sutton-bonington"),

    # --- Coastal Extras ---
    ("tyne-and-wear", "00251", "newcastle"),
    ("avon", "00313", "bristol"),
        # --- Extra 5 (to push beyond 50 safely) ---
    ("cheshire", "00348", "shawbury"),
    ("norfolk", "00411", "marham"),
    ("suffolk", "00420", "mildenhall"),
    ("gloucestershire", "00512", "pershore"),
    ("northamptonshire", "00488", "church-lawford"),

]


# ==============================
# DOWNLOAD LOOP
# ==============================

print("\nStarting downloads...\n")

for county, src_id, station_name in STATIONS:

    station_folder = f"{src_id}_{station_name}"
    station_dir = os.path.join(OUTPUT_DIR, station_folder)
    os.makedirs(station_dir, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):

        save_path = os.path.join(station_dir, f"{year}.csv")

        # Skip existing
        if os.path.exists(save_path):
            continue

        filename = (
            f"midas-open_uk-daily-weather-obs_dv-202507_"
            f"{county}_{src_id}_{station_name}_qcv-1_{year}.csv"
        )

        url = f"{BASE_URL}/{county}/{station_folder}/qc-version-1/{filename}"

        print(f"Downloading {station_folder} {year}...")

        try:
            response = requests.get(url, headers=HEADERS, timeout=30)

            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
            elif response.status_code == 404:
                continue
            else:
                print(f"Error {station_folder} {year}: {response.status_code}")

        except Exception as e:
            print(f"Connection error {station_folder} {year}: {e}")

print("\nDownload process completed.")
