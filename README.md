# AutoJudge – Predicting Programming Problem Difficulty
AutoJudge is a machine learning project where the goal is to predict the difficulty of programming problems automatically.
Usually, online coding platforms decide problem difficulty based on human judgment and user feedback. In this project, we try to predict the difficulty only from the problem text.

The system predicts:

Difficulty level: Easy / Medium / Hard

Difficulty score: a numerical value

# What This Project Does
Given the text of a programming problem (description, input, and output), the model tries to understand how complex the problem is and predicts:

The difficulty category

A difficulty score

This project mainly focuses on text processing and basic machine learning models.

# Dataset Information
The dataset used in this project contains the following fields:

Title

Problem description

Input description

Output description

Problem class (Easy / Medium / Hard)

Problem score (numerical value)

#  Approach

Combine all text fields into a single text input

Convert text to numerical features using TF-IDF

Train:

A classification model for difficulty class

A regression model for difficulty score

# How to Run the Project

1.Create and activate a virtual environment

2.Install dependencies

3.Run the project:

  python app.py

# Web Interface

Users can paste a problem’s description, input, and output and get:

Predicted difficulty class

Predicted difficulty score

# Tools Used

Python

Scikit-learn

TF-IDF

Flask / Streamlit

