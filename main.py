import faker
import random
import logging
import datetime
import json
import re
import numpy as np
import time
import holidays
from dataclasses import dataclass, field
from itertools import count
from google.cloud import bigquery
from google.api_core.exceptions import Conflict, NotFound
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models


# logging helper
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
total_rows_streamed = 0

# Configuration and Client (replace with your project details)

PROJECT_ID = "gemini-looker-demo-dataset"
DATASET_ID = "cloudbites_dev"
DATASET_LOCATION = "US"
BATCH_SIZE = 25000
COMPANY_CREATION_DATE = datetime.datetime(2020, 1, 1)
MINUTES_IN_HOUR = 60

# BigQuery Client

client = bigquery.Client(project=PROJECT_ID)

# Faker (replace with your desired locale)
fake = faker.Faker()

# ID Generation
company_id_gen = count(1)  # Starts at 1
franchisee_id_gen = count(1)
restaurant_id_gen = count(1)
order_id_gen = count(1)
order_item_id_gen = count(1)
menu_id_gen = count(1)

# Prompts

FRANCHISEE_GENERATION_PROMPT = """
Please generate exactly {number} company names. Not more names, not less, and return in below json format. Every Company name is one json object, no duplicates possible.

- Think about company names suitable for a holding company operating multiple burger places.
- Nothing that has any similarity to McDonalds or Burger King.
- Be creative and funny
- I just want the list of names.

Sample JSON Response: [name: "name"]
"""

MENU_GENERATION_PROMPT = """
Please generate a restaurant menu with {number} items in total. Not more, not less, and return in below json format. Every menu item is one json object, no duplicates possible.

- Restaurant serves primarily burgers, the company name is Cloudbites.
- Also include typical sides and beverages that go well with Burgers and are traditional.
- Include item_name, item_price, item_size. Nothing more, nothing less.
- Distinguish on sizes between small, medium, and large, but not necessarily for every individual item, only where it makes sense.
- Make sure to include some kind of Milkshake.
- Nothing that has any similarity to McDonalds or Burger King.
- Be creative and funny

Sample JSON Response: [name: "name", price: "price", size: "value1"] Important: no values should be Null.

"""

#


def generate_llm(prompt: str, number: int):
    vertexai.init(project="gemini-looker-demo-dataset", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-001",
    )

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    formatted_prompt = prompt.format(number=number)

    response = model.generate_content(
        formatted_prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    try:
        str = (
            re.search(r"```json\n(.*)\n```", response.text, re.DOTALL).group(1).strip()
        )
    except:
        return generate_llm(prompt, number)

    response = json.loads(str)

    return response


def generate_menu(data):
    menu_data = data
    # menu_data = [
    #     {
    #         "item_name": "test",
    #         "item_price": 0.34,
    #         "item_size": "test",
    #     },
    #     {
    #         "item_name": "test",
    #         "item_price": 0.28,
    #         "item_size": "test",
    #     },
    # ]
    logging.info(f"generating 50 menu items with Gemini")
    menu_items = []
    for item in menu_data:
        menu = Menu(
            menu_id=next(menu_id_gen),
            menu_item_name=item["item_name"],
            menu_item_price=item["item_price"],
            menu_item_size=item["item_size"],
        )
        menu_items.append(menu.__dict__)
    stream_data("menu", menu_items)
    return [Menu(**item) for item in menu_items]


class DataUtil:
    seasonal_weights = {
        1: 0.11,  # January
        2: 0.09,  # February
        3: 0.08,  # March
        4: 0.11,  # April
        5: 0.11,  # May
        6: 0.11,  # June
        7: 0.12,  # July
        8: 0.10,  # August
        9: 0.1,  # September
        10: 0.08,  # October
        11: 0.09,  # November
        12: 0.1,  # December
    }

    def created_at(start_date: datetime.datetime) -> datetime.datetime:
        end_date = datetime.datetime.now()
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        if days_between_dates <= 1:
            days_between_dates = 2

        # # Define parameters for the probability function
        # midpoint = (days_between_dates // 100) * 20
        # steepness = 0.002

        # Calculate probability using the calculate_probability helper function
        # probabilities = [
        #     calculate_probability(i, midpoint, steepness)
        #     for i in range(1, days_between_dates)
        # ]
        # random_number_of_days = random.choices(
        #     range(1, days_between_dates), weights=probabilities
        # )[0]
        random_number_of_days = random.choices(range(1, (days_between_dates // 3)))[0]
        created_at = (
            start_date
            + datetime.timedelta(days=random_number_of_days)
            + datetime.timedelta(minutes=random.randrange(MINUTES_IN_HOUR * 19))
        )
        return created_at

    def child_created_at_seasonality(date: datetime.datetime) -> datetime.datetime:
        time_between_dates = datetime.datetime.now() - date
        days_between_dates = time_between_dates.days
        if days_between_dates <= 1:
            days_between_dates = 2

        # --- Generate a random date within the range ---
        random_number_of_days = random.randrange(1, days_between_dates)
        random_date = date + datetime.timedelta(days=random_number_of_days)

        # Get month of the year (1-12) for the random date

        random_month = random_date.month

        # Determine which country holidays apply
        # if self.parent.country_code == "USA":
        #     holidays_list = holidays.country_holidays("US")
        # elif self.parent.country_code == "JPN":
        #     holidays_list = holidays.country_holidays("JP")
        # elif self.parent.country_code == "GBR":
        #     holidays_list = holidays.country_holidays("GB")

        holidays_list = holidays.country_holidays("US")

        # Check if random date falls within public holiday range ()

        for day_offset in range(-4, 4):
            holiday_date = random_date + datetime.timedelta(days=day_offset)
            if holiday_date in holidays_list:
                adjusted_weight = DataUtil.seasonal_weights[random_month] * 1.5
                break
            else:
                adjusted_weight = DataUtil.seasonal_weights[random_month]

        # Apply acceptance probability based on scaled seasonal weight
        acceptance_probability = random.random()
        if acceptance_probability > adjusted_weight:
            return DataUtil.child_created_at_seasonality(date)  # Retry if not accepted
        else:
            return random_date

    def normal_distribution(mean, std_dev):
        return max(
            0,
            round(
                np.random.normal(
                    mean,
                    std_dev,
                )
            ),
        )

    def number_of_orders(date: datetime.datetime, daily_orders: int) -> int:
        start_date = date
        end_date = datetime.datetime.now()
        time_between_days = (end_date - start_date).days
        orders = time_between_days * daily_orders
        return orders


# Data Classes
@dataclass
class Menu:
    menu_id: int
    menu_item_name: str = None
    menu_item_price: str = None
    menu_item_size: str = None

    def __post_init__(self):
        self.menu_item_price = float(self.menu_item_price)


@dataclass
class Company:
    company_name: str
    company_creation_date: datetime.datetime
    company_id: int = field(default_factory=lambda: next(company_id_gen))


@dataclass
class Franchisee:
    company_id: int
    franchisee_name: str
    franchisee_since_datetime: str
    franchisee_city: str
    franchisee_state: str
    franchisee_id: int = field(default_factory=lambda: next(franchisee_id_gen))


@dataclass
class Restaurant:
    franchisee_id: int
    restaurant_opening_date: str
    restaurant_capacity_seats: int
    restaurant_city: str
    restaurant_state: str
    restaurant_id: int = field(default_factory=lambda: next(restaurant_id_gen))


@dataclass
class Order:
    restaurant_id: int
    order_datetime: str
    order_completion_datetime: str = None
    order_id: int = field(default_factory=lambda: next(order_id_gen))
    order_dine_in: bool = random.choices([True, False], [0.67, 0.33])[0]

    def __post_init__(self):
        self.order_completion_datetime = self.order_datetime + datetime.timedelta(
            minutes=random.randrange(2, 60)
        )


@dataclass
class OrderItem:
    order_id: int
    menu_id: int
    order_item_price: float
    order_item_size: str
    order_item_name: str
    order_item_quantity: int = None
    order_item_total_price: float = None
    order_item_id: int = field(default_factory=lambda: next(order_item_id_gen))

    def __post_init__(self):
        self.order_item_quantity = random.randint(1, 4)
        self.order_item_total_price = self.order_item_price * self.order_item_quantity


# Dataset and Table Creation with Error Handling
def create_dataset_and_tables():
    try:
        client.get_dataset(f"{client.project}.{DATASET_ID}")  # Check if dataset exists
        print(f"Dataset {DATASET_ID} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(f"{client.project}.{DATASET_ID}")
        dataset.location = DATASET_LOCATION  # Replace with your desired location
        client.create_dataset(dataset)
        print(f"Created dataset {DATASET_ID}")
    schemas = {
        "menu": [
            bigquery.SchemaField("menu_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("menu_item_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("menu_item_price", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("menu_item_size", "STRING", mode="REQUIRED"),
        ],
        "company": [
            bigquery.SchemaField("company_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("company_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("company_creation_date", "TIMESTAMP", mode="REQUIRED"),
        ],
        "franchisee": [
            bigquery.SchemaField("franchisee_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("company_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("franchisee_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "franchisee_since_datetime", "TIMESTAMP", mode="REQUIRED"
            ),
            bigquery.SchemaField("franchisee_city", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("franchisee_state", "STRING", mode="REQUIRED"),
        ],
        "restaurant": [
            bigquery.SchemaField("restaurant_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("franchisee_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField(
                "restaurant_capacity_seats", "INTEGER", mode="REQUIRED"
            ),
            bigquery.SchemaField("restaurant_city", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("restaurant_state", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "restaurant_opening_date", "TIMESTAMP", mode="REQUIRED"
            ),
        ],
        "order": [
            bigquery.SchemaField("order_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("restaurant_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("order_datetime", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField(
                "order_completion_datetime", "TIMESTAMP", mode="REQUIRED"
            ),
            bigquery.SchemaField("order_dine_in", "BOOLEAN", mode="REQUIRED"),
        ],
        "order_item": [
            bigquery.SchemaField("order_item_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("order_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("order_item_quantity", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("order_item_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("order_item_price", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("order_item_total_price", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("order_item_size", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("menu_id", "INTEGER", mode="REQUIRED"),
        ],
    }

    for table_name, schema in schemas.items():
        table_id = f"{client.project}.{DATASET_ID}.{table_name}"

        try:
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Created table {table_id}")
        except Conflict:
            print(f"Table {table_id} already exists. Skipping creation.")


# Streaming Function
def stream_data(table_name, data_list):
    global total_rows_streamed
    table_id = f"{client.project}.{DATASET_ID}.{table_name}"

    max_retries = 3  # You can adjust the number of retries
    retry_delay = 2  # Seconds to wait between retries

    for attempt in range(max_retries + 1):

        try:
            # Convert datetime objects to ISO format strings before streaming
            for data in data_list:
                for key, value in data.items():
                    if isinstance(value, datetime.datetime):
                        data[key] = value.isoformat()
            errors = client.insert_rows_json(table_id, data_list)
            if errors:
                print(
                    f"Encountered errors while inserting rows into {table_id}: {errors}"
                )
            else:
                total_rows_streamed += len(data_list)  # increase counter
                logging.info(f"Streamed {len(data_list)} rows to {table_id}")
                break
        except NotFound:
            if attempt < max_retries:
                logging.warning(
                    f"Table {table_id} not found. Retrying in {retry_delay} seconds... (Attempt {attempt + 1})"
                )
                time.sleep(retry_delay)
            else:
                logging.error(
                    f"Table {table_id} not found after {max_retries} retries. Aborting."
                )
                raise  # Re-raise the exception after max retries

        except Exception as e:  # Catch all other exceptions to avoid blocking
            print(f"Error streaming to {table_name}: {e}")


# Generation Logic
def generate_data(company_name):
    # Generate dataset and tables
    create_dataset_and_tables()
    logging.info("Starting data generation and streaming..")
    # time.sleep(60)
    # Set variables

    # menu_items = generate_menu(
    #     [
    #         {
    #             "item_name": "test",
    #             "item_price": 0.34,
    #             "item_size": "test",
    #         },
    #         {
    #             "item_name": "test",
    #             "item_price": 0.28,
    #             "item_size": "test",
    #         },
    #     ]
    # )

    # franchisee_names = [{"name": "test"}, {"name": "test"}]
    num_of_menu_items = 50
    num_of_franchisees = 4
    # num_of_orders = 50000
    # num_of_orders = DataUtil.number_of_orders(COMPANY_CREATION_DATE, 75)

    # Generate menu
    menu_items = generate_menu(generate_llm(MENU_GENERATION_PROMPT, num_of_menu_items))
    company = Company(
        company_name=company_name, company_creation_date=COMPANY_CREATION_DATE
    )

    # Generate franchisee names
    franchisee_names = generate_llm(FRANCHISEE_GENERATION_PROMPT, num_of_franchisees)
    # Data Generation
    company_data = [company.__dict__]

    franchisee_data = []
    restaurant_data = []
    order_data = []
    order_item_data = []

    for f in franchisee_names:
        franchisee = Franchisee(
            company_id=company.company_id,
            franchisee_since_datetime=DataUtil.created_at(
                company.company_creation_date
            ),
            franchisee_name=f["name"],
            franchisee_city=fake.city(),
            franchisee_state=fake.state(),
        )
        franchisee_data.append(franchisee.__dict__)

        num_of_restaurants = DataUtil.normal_distribution(5, 1)
        for _ in range(num_of_restaurants):
            restaurant = Restaurant(
                franchisee_id=franchisee.franchisee_id,
                restaurant_opening_date=DataUtil.created_at(
                    franchisee.franchisee_since_datetime
                ),
                restaurant_city=fake.city(),
                restaurant_state=fake.state(),
                restaurant_capacity_seats=random.randint(1, 100),
            )
            restaurant_data.append(restaurant.__dict__)

            num_of_orders = DataUtil.number_of_orders(
                restaurant.restaurant_opening_date, 65
            )
            # num_of_orders = 10000
            for _ in range(num_of_orders):
                order = Order(
                    restaurant.restaurant_id,
                    order_datetime=DataUtil.child_created_at_seasonality(
                        restaurant.restaurant_opening_date
                    ),
                )
                order_data.append(order.__dict__)
                num_of_order_items = DataUtil.normal_distribution(3, 1)
                for _ in range(num_of_order_items):
                    selected_menu_item = random.choice(menu_items)
                    order_item = OrderItem(
                        menu_id=selected_menu_item.menu_id,
                        order_id=order.order_id,
                        order_item_price=selected_menu_item.menu_item_price,
                        order_item_name=selected_menu_item.menu_item_name,
                        order_item_size=selected_menu_item.menu_item_size,
                    )
                    order_item_data.append(order_item.__dict__)

                    if len(order_item_data) >= BATCH_SIZE:
                        stream_data("order_item", order_item_data)
                        order_item_data = []

                if len(order_data) >= BATCH_SIZE:
                    stream_data("order", order_data)
                    order_data = []

            if len(restaurant_data) >= BATCH_SIZE:
                stream_data("restaurant", restaurant_data)
                restaurant_data = []

        if len(franchisee_data) >= BATCH_SIZE:
            stream_data("franchisee", franchisee_data)
            franchisee_data = []

    # Stream any remaining data
    if franchisee_data:
        stream_data("franchisee", franchisee_data)
    if restaurant_data:
        stream_data("restaurant", restaurant_data)
    if order_data:
        stream_data("order", order_data)
    if order_item_data:
        stream_data("order_item", order_item_data)

    stream_data("company", company_data)

    logging.info(f"Finished streaming data. Total rows streamed: {total_rows_streamed}")


if __name__ == "__main__":
    generate_data(
        company_name="Cloud Bites",
    )
