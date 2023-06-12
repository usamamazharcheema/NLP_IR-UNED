import argparse
import subprocess


def run():


    #data = "default"
    data = "2022_2b"

    subprocess.call(["python", "evaluation/scorer/main.py", data])


if __name__ == "__main__":
    run()