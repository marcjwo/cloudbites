import base64
import dataclasses
import itertools
import datetime
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import holidays
from google.cloud import bigquery
import random
from faker import Faker
import typing
import logging
import re
import json

fake = Faker()


company = []
cities = ["Los Angeles", "San Bernadino", "New York City", "Chicago"]
franchisee = []
restaurant = []
order = []
order_item = []
menu = []
company_creation_date = datetime.datetime(2020, 1, 1)
number_of_franchisees = 1
target_gcp_project = "gemini-looker-demo-dataset"
target_bq_dataset = "cloudbites"


### Prompts and Gemini bkts

franchisee_generation_prompt = """
Please generate exactly {number} company names. Not more names, not less.

- Think about company names suitable for a holding company operating multiple burger places.
- Nothing that has any similarity to McDonalds or Burger King.
- Be creative and funny
- I just want the list of names.

Example:
["Out and About"]
["Sizzling Inc", "Burgermeister"]
"""

menu_generation_prompt = """
Please generate a restaurant menu with {number} items in total. Not more, not less, and return in below json format. Every menu item is one json object, no duplicates possible.

- Restaurant surfs primarily burgers, the company name is Cloudbites.
- Also include typical sides and beverages that go well with Burgers and are traditional.
- Include item_name, item_price, item_size. Distinguish on sizes between small, medium, and large.
- Nothing that has any similarity to McDonalds or Burger King.
- Be creative and funny

Sample JSON Response: [name: "name", price: "price", size: "value1"]

"""


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


franchisee_data = generate_llm(franchisee_generation_prompt, 5)
menu_data = generate_llm(menu_generation_prompt, 5)

print(franchisee_data)
print(menu_data)

###


class DataUtil:
    MINUTES_IN_HOUR = 60
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

    @staticmethod
    def load_to_bq(
        table_name: str,
        table_data: list,
        project_id: str,
        dataset_id: str,
        schema: list = None,
    ) -> None:
        client = bigquery.Client()
        if schema is None:
            job_config = bigquery.LoadJobConfig(
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
        else:
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
        no_touch_tables = ["menu", "franchisee"]
        if table_name in no_touch_tables:
            table = client.get_table(f"{project_id}.{dataset_id}.{table_name}")
            if table.num_rows > 0:
                print(f"{table_name} table already exists and does not need a rewrite")
            else:
                table_id = f"{project_id}.{dataset_id}.{table_name}"
                job = client.load_table_from_json(
                    table_data, table_id, job_config=job_config
                )
                job.result()
                if job.errors:
                    logging.error(job.errors)
                else:
                    logging.info(
                        f"loaded {table_name} successfully with {len(table_data)} rows"
                    )
        else:
            table_id = f"{project_id}.{dataset_id}.{table_name}"
            job = client.load_table_from_json(
                table_data, table_id, job_config=job_config
            )
            job.result()
            if job.errors:
                logging.error(job.errors)
            else:
                logging.info(
                    f"loaded {table_name} successfully with {len(table_data)} rows"
                )

    @staticmethod
    def number_of_orders(date: datetime.datetime, daily_orders: int) -> int:
        start_date = date
        end_date = datetime.datetime.now()
        time_between_days = (end_date - start_date).days
        orders = time_between_days * daily_orders
        return orders

    @staticmethod
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
        random_number_of_days = random.choices(range(1, days_between_dates))[0]
        created_at = (
            start_date
            + datetime.timedelta(days=random_number_of_days)
            + datetime.timedelta(
                minutes=random.randrange(DataUtil.MINUTES_IN_HOUR * 19)
            )
        )
        return created_at

    @staticmethod
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

    @staticmethod
    def generate_random_location():
        loc = fake.local_latlng(coords_only=False)
        location = loc[0] + "," + loc[1]
        city = loc[2]

        return city, location


@dataclasses.dataclass(slots=True)
class Menu:
    logging.info("generating Menu information")
    item_name: str = dataclasses.field()
    item_price: int = dataclasses.field()
    item_size: str = dataclasses.field()
    menu_id: int = dataclasses.field(default_factory=itertools.count(start=1).__next__)

    def __post_init__(self):
        num_of_menu_items = 50
        menu_items = DataUtil.generate_llm(menu_generation_prompt, num_of_menu_items)
        for m in menu_items:
            self.item_name = m["item_name"]
            self.item_price = m["item_price"]
            self.item_size = m["item_size"]


@dataclasses.dataclass(slots=True)
class Company:
    logging.info("generating Company information")
    company_name: str = dataclasses.field()
    company_id: int = dataclasses.field(
        default_factory=itertools.count(start=1).__next__
    )

    def __post_init__(self):
        # self.company_name = "Cloudbites"
        num_of_franchisees = 2
        logging.info("generating franchisee names using Gemini")
        franchisee_names = [
            {"name": "The Patty Palace"},
            {"name": "Burgeropolis Holdings"},
        ]
        # franchisee_names = DataUtil.generate_llm(
        #     franchisee_generation_prompt, num_of_franchisees
        # )
        # logging.info(f"{franchisee_names} generated")
        for f in franchisee_names:
            logging.info(f"generating franchisee {f} information")
            franchisee.append(
                dataclasses.asdict(Franchisee(company=self, name=f["name"]))
            )
        # for _ in range(num_of_franchisees):
        #     franchisee.append(dataclasses.asdict(Franchisee(company=self)))


@dataclasses.dataclass(slots=True)
class Franchisee:
    name: str
    franchisee_id: int = dataclasses.field(
        default_factory=itertools.count(start=1).__next__
    )
    company_id: int = dataclasses.field(init=False)
    location: str = dataclasses.field(init=False)
    city: str = dataclasses.field(init=False)
    # address: str = dataclasses.field(init=False)
    # latitude: float = dataclasses.field(init=False)
    # longitude: float = dataclasses.field(init=False)
    franchisee_since: datetime.datetime = dataclasses.field(init=False)
    company: dataclasses.InitVar[typing.Any] = None

    def __post_init__(self, company=None):
        self.company_id = company.company_id
        # self.name = DataUtil.generate_llm("franchisee_generation_prompt")
        city, location = DataUtil.generate_random_location()
        self.city = city
        self.location = location
        # random.choices(cities, weights=[0.15, 0.25, 0.2, 0.4])[0]
        # address = DataUtil.generate_address_in_city(self.city)
        # self.address = address[0]
        # self.latitude = address[1]
        # self.longitude = address[2]
        self.franchisee_since = DataUtil.created_at(company_creation_date)
        num_of_restaurants = max(1, round(np.random.normal(10, 5)))
        for _ in range(num_of_restaurants):
            logging.info(f"generating restaurant no {_} information")
            restaurant.append(dataclasses.asdict(Restaurant(franchisee=self)))


@dataclasses.dataclass(slots=True)
class Restaurant:
    logging.info("generating restaurant information")
    restaurant_id: int = dataclasses.field(
        default_factory=itertools.count(start=1).__next__
    )
    franchisee_id: int = dataclasses.field(init=False)
    size_seats: str = dataclasses.field(init=False)
    city: str = dataclasses.field(init=False)
    location: str = dataclasses.field(init=False)
    # address: str = dataclasses.field(init=False)
    # latitude: float = dataclasses.field(init=False)
    # longitude: float = dataclasses.field(init=False)
    opening_date: datetime.datetime = dataclasses.field(init=False)
    franchisee: dataclasses.InitVar[typing.Any] = None

    def __post_init__(self, franchisee=None):
        self.franchisee_id = franchisee.franchisee_id
        self.size_seats = max(1, round(np.random.normal(45, 15)))
        city, location = DataUtil.generate_random_location()
        self.city = city
        self.location = location
        # address = DataUtil.generate_address_in_city(franchisee.city)
        # self.address = address[0]
        # self.latitude = address[1]
        # self.longitude = address[2]
        self.opening_date = DataUtil.child_created_at_seasonality(
            franchisee.franchisee_since
        )
        # num_of_orders = DataUtil.number_of_orders(self.opening_date, 1)
        num_of_orders = 10
        for _ in range(num_of_orders):
            logging.info(f"generating order no {_}")
            order.append(dataclasses.asdict(Order(restaurant=self)))


@dataclasses.dataclass(slots=True)
class Order:
    logging.info("generating order information")
    order_id: int = dataclasses.field(default_factory=itertools.count(start=1).__next__)
    restaurant_id: int = dataclasses.field(init=False)
    # customer_id: int = dataclasses.field(init=False)
    order_datetime: datetime.datetime = dataclasses.field(init=False)
    order_completion_datetime: datetime.datetime = dataclasses.field(init=False)
    dine_in: bool = dataclasses.field(init=False)
    restaurant: dataclasses.InitVar[typing.Any] = None

    def __post_init__(self, restaurant=None):
        # self.parent = customer
        # self.location_id = random.choice([d["location_id"] for d in LOCATION_DATA])
        self.restaurant_id = restaurant.restaurant_id
        self.order_datetime = DataUtil.child_created_at_seasonality(
            restaurant.opening_date
        )
        self.order_completion_datetime = self.order_datetime + datetime.timedelta(
            minutes=random.randrange(2, 60)
        )
        self.dine_in = random.choices([True, False], weights=[0.6, 0.4])[0]
        # self.customer_id = random.choice([d["customer_id"] for d in CUSTOMER_DATA])

        # randomly generate number of items in order
        num_of_items = random.choices(
            population=[1, 2, 3, 4], weights=[0.7, 0.2, 0.05, 0.05]
        )[0]
        # for _ in range(num_of_items):
        #     order_items.append(dataclasses.asdict(OrderItem(order=self)))


# @dataclasses.dataclass(slots=False)
# class OrderItem:
#     logging.info("generating order item")
#     order_item_id: int = itertools.count(start=1).__next__()
#     order_id: int = dataclasses.field(init=False)
#     menu_id: int = dataclasses.field(init=False)
#     quantity: int = dataclasses.field(init=False)
#     item_size: str = dataclasses.field(init=False)
#     item_price: float = dataclasses.field(init=False)
#     item_total: float = dataclasses.field(init=False)
#     order: dataclasses.InitVar[typing.Any] = None

#     def __post_init__(self, order=None):
#         self.order_id = order.order_id
#         self.menu_id = random.choice([d["menu_id"] for d in MENU_DATA])
#         self.quantity = random.randrange(1, 3)
#         # TODO: add size, price, and total from the menu
#         try:
#             self.item_size = next(
#                 row["item_size"] for row in MENU_DATA if row["menu_id"] == self.menu_id
#             )
#         except StopIteration:
#             self.item_size = None
#         try:
#             self.item_price = next(
#                 row["item_price"] for row in MENU_DATA if row["menu_id"] == self.menu_id
#             )
#         except StopIteration:
#             self.item_price = None
#         self.item_total = self.quantity * self.item_price


def main():
    num_of_menu_items = 50
    logging.info(f"generating {num_of_menu_items} with Gemini")
    menu_data = DataUtil.generate_llm(menu_generation_prompt, num_of_menu_items)
    for m in menu_data:
        logging.info(f"appending {m} to menu")
        menu.append(
            dataclasses.asdict(
                Menu(
                    item_name=m["item_name"],
                    item_price=m["item_price"],
                    item_size=m["item_size"],
                )
            )
        )
    company.append(dataclasses.asdict(Company(company_name="Cloudbites")))
    # for name in franchisee_names:
    #     print(name)
    #     logging.info(f"generating franchisee {name}")
    #     franchisee.append(dataclasses.asdict(Franchisee(name=name)))

    for f in franchisee:
        f["franchisee_since"] = f["franchisee_since"].isoformat()
    for r in restaurant:
        r["opening_date"] = r["opening_date"].isoformat()
    for o in order:
        o["order_datetime"] = o["order_datetime"].isoformat()
        o["order_completion_datetime"] = o["order_completion_datetime"].isoformat()
    table_name = ["company", "franchisee", "restaurant", "order", "menu"]
    table_data = [company, franchisee, restaurant, order, menu]
    # table_schema = [
    #     [
    #         bigquery.SchemaField("company_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("company_name", "STRING", mode="REQUIRED"),
    #     ],
    #     [
    #         bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
    #         bigquery.SchemaField("franchisee_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("company_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("city", "STRING", mode="REQUIRED"),
    #         # bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
    #         bigquery.SchemaField("location", "STRING", mode="REQUIRED"),
    #         # bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
    #         # bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
    #         bigquery.SchemaField("franchisee_since", "TIMESTAMP", mode="REQUIRED"),
    #     ],
    #     [
    #         bigquery.SchemaField("restaurant_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("franchisee_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("size_seats", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("city", "STRING", mode="REQUIRED"),
    #         # bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
    #         bigquery.SchemaField("location", "STRING", mode="REQUIRED"),
    #         # bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
    #         # bigquery.SchemaField(
    #         #     "latitude",
    #         #     "FLOAT",
    #         #     mode="REQUIRED",
    #         #     description="Latitude of the restaurant",
    #         # ),
    #         # bigquery.SchemaField(
    #         #     "longitude",
    #         #     "FLOAT",
    #         #     mode="REQUIRED",
    #         #     description="Longitude of the restaurant",
    #         # ),
    #         bigquery.SchemaField(
    #             "opening_date",
    #             "TIMESTAMP",
    #             mode="REQUIRED",
    #             description="Opening date of the restaurant",
    #         ),
    #     ],
    #     [
    #         bigquery.SchemaField("order_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("restaurant_id", "INTEGER", mode="REQUIRED"),
    #         # bigquery.SchemaField("customer_id", "INTEGER", mode="REQUIRED"),
    #         bigquery.SchemaField("order_datetime", "TIMESTAMP", mode="REQUIRED"),
    #         bigquery.SchemaField(
    #             "order_completion_datetime", "TIMESTAMP", mode="REQUIRED"
    #         ),
    #         bigquery.SchemaField("dine_in", "BOOLEAN", mode="REQUIRED"),
    #     ],
    # ]
    # for name, data, schema in list(zip(table_name, table_data, table_schema)):
    for name, data in list(zip(table_name, table_data)):
        logging.info(f"writing {name} to BQ...")
        DataUtil.load_to_bq(
            table_name=name,
            table_data=data,
            # schema=schema,
            project_id=target_gcp_project,
            dataset_id=target_bq_dataset,
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
