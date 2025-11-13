# GenDebate – Multi-persona Ne Zha Debate Generator

This script generates multi-person debates about the **“Ne Zha” (2019)** trailer using the OpenAI API, then auto-analyzes each debate.

## 1. Requirements

- Python 3.9+
- An OpenAI API key
- Python package:




# Configuration guide – How to customize the debate generator

This document explains how **anyone** can customize the script `Chat_gen_API_OPENIA.py` without touching the core logic.

All configuration is done through a few **constants at the top of the file** and the **PERSONAS** list.

---

## 1. Basic configuration

At the top of the script, you’ll find:

python
OPENAI_API_KEY = ""
MODEL          = "gpt-4o-mini"
TARGET_TURNS   = 20
SESSIONS       = 3
DEBATE_PATH    = "DEBATE"
SLEEP_BETWEEN  = 1.0
SEED           = 123
REACT_MIN_PCT  = 0.85
MAX_ATTEMPTS   = 3

ANALYSIS_MODEL   = "gpt-4o-mini"
ANALYSIS_MAXTOK  = 2000
ANALYSIS_OUTFILE = Path("ANALYSES/analysis_last_session.md")
APPEND_ANALYSIS_IN_DEBATE = True

## You need API key here : https://platform.openai.com/api-keys

```bash
pip install --upgrade openai


