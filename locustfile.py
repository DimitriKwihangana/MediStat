from locust import HttpUser, task, between
import random
import json

class LoadTestUser(HttpUser):
    wait_time = between(1, 5)  # Simulate a delay between requests

    @task(5)  
    def predict(self):
        self.client.post(
            "/predict/",
            json={
                "baseline_value": 120,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0,
                "light_decelerations": 0.0,
                "prolongued_decelerations": 0.0,
                "abnormal_short_term_variability": 0.0,
                "mean_value_of_short_term_variability": 0.5,
                "percentage_of_time_with_abnormal_long_term_variability": 2.0,
                "histogram_width": 70,
                "histogram_min": 62,
                "histogram_max": 135,
                "histogram_number_of_peaks": 2,
                "histogram_number_of_zeroes": 0,
                "histogram_mode": 120,
                "histogram_median": 120,
                "histogram_tendency": 1
            }
        )

    @task(2)  # Assign moderate weight to `retrain`
    def retrain(self):
        # Simulate uploading a CSV file for retraining
        with open("sample_data.csv", "rb") as file:
            self.client.post(
                "/retrain/",
                files={"file": ("sample_data.csv", file, "text/csv")}
            )

    @task(1)  # Assign lower weight to `fine_tune`
    def fine_tune(self):
        # Simulate uploading a CSV file and providing the `epochs` parameter
        with open("sample_data.csv", "rb") as file:
            self.client.post(
                "/fine_tune/",
                files={"file": ("sample_data.csv", file, "text/csv")},
                data={"epochs": random.randint(5, 15)}  # Random number of epochs
            )
