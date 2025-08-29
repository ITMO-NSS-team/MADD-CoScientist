import streamlit as st
from dotenv import load_dotenv

from MADD.frontend import chat, init_page, side_bar
from MADD.frontend.utils import start_cleanup_thread

load_dotenv('config.env')


if __name__ == '__main__':
    start_cleanup_thread()
    init_page()
    side_bar()
    chat()


