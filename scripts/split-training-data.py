import os
import json
import typer
import logging


from typing import Annotated

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = typer.Typer()

@app.command()
def main(input_dir: str, output_dir: str):


if __name__ == "__main__":
    app()
