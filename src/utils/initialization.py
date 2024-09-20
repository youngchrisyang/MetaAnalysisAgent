from dotenv import load_dotenv, find_dotenv

def initialize_env():
    _ = load_dotenv(find_dotenv()) # read local .env file
    