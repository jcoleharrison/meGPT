# GPT-Me

## Overview
This project is to ...

## Setup
- Run extract.py pointing at your iMessage db (you may have to make a copy in a new location for this to run)
    - (WARNING): Be extremely careful moving around your iMessage data. There is likely much information you wouldn't want to be stolen or accidentally free to the public :)
- Run prep.py to get data prepped and tokenized into a ready-to-train format

```
pip install --upgrade transformers accelerate bitsandbytes peft datasets

```