# CV&DS Admission – Dikshant Bikram Thapa

Application material for the NHL Stenden Master Computer Vision & Data Science (2027-2028).

## Contents
- `assignments/debugging_solved.ipynb` – the programme's notebook with minimal `# FIX` edits and an explanation after each exercise
- `assignments/debugging_exercises.py` – exercises 1-3 (bugs explained in comments, self-check included)
- `personal-code/image_basics.py` – my own Python code: grayscale loading, mean blur (loops), threshold, histogram; tests in `test_image_basics.py`
- `assignments/gan_exercise.py` – exercise 4, GAN on MNIST (structural + cosmetic bug fixed)

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exercises 1-3, personal code
pip install torch torchvision            # exercise 4 only
python assignments/debugging_exercises.py
cd personal-code && python test_image_basics.py && python image_basics.py
```
