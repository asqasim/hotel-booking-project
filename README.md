Hotel Booking Cancellation Prediction
IDS f24 – Introduction to Data Science Semester Project

Project Overview
----------------
Hotel booking cancellations are a major challenge for the hospitality industry, leading to revenue loss and inefficient resource planning.
This project analyzes hotel booking data and predicts whether a reservation will be canceled or checked-in using machine learning.

The project demonstrates the complete data science lifecycle:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Interactive deployment using Streamlit

Objectives
----------
- Identify key factors influencing booking cancellations
- Perform detailed exploratory data analysis
- Train and evaluate a Random Forest classifier
- Build an interactive web application for real-time predictions

Project Structure
-----------------
hotel-booking-project/
├── data/
│   └── hotel_bookings.csv
├── models/
│   └── hotel_cancellation_model.pkl
├── src/
│   └── app.py
├── requirements.txt
└── README.txt

Dataset
-------
- Approximately 119,000 hotel bookings
- Target variable: is_canceled
- Features include lead time, ADR, market segment, deposit type, country,
  previous cancellations, and special requests

Machine Learning Model
----------------------
- Algorithm: Random Forest Classifier
- Handles non-linear relationships and mixed data types
- Outputs probability of booking cancellation

Streamlit Application
---------------------
The app includes:
- Home page with dataset overview
- Interactive EDA visualizations
- Prediction form with clean UI controls
- Highlighted prediction results with probabilities
- Key takeaways section summarizing insights

Installation & Usage
--------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run src/app.py

Key Takeaways
-------------
- Lead time strongly affects cancellation likelihood
- Previous cancellations increase risk
- Deposit type and market segment matter
- Random Forest performs well for this task

Course Information
------------------
Course: Introduction to Data Science (IDS f24)
Project Type: Semester Project
