"""
Locust load testing script for Diabetes Readmission API
Simulates users sending predict requests to the API
"""

from locust import HttpUser, task, between
import random


class DiabetesAPITestUser(HttpUser):
    """
    Simulates a user making prediction requests to the Diabetes API
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Health check to ensure API is available
        self.client.get("/health")
    
    @task(3)
    def predict_readmission(self):
        """
        Main task: Send a prediction request with realistic patient data
        Weight: 3 (most common task)
        """
        # Sample patient features for diabetes readmission prediction
        # Using realistic data based on the API schema
        patient_data = {
            "features": {
                "race": random.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]),
                "gender": random.choice(["Male", "Female", "Unknown/Invalid"]),
                "age": random.choice(["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]),
                "weight": random.choice(["?", "[0-25)", "[25-50)", "[50-75)", "[75-100)", 
                                       "[100-125)", "[125-150)", "[150-175)", "[175-200)"]),
                "admission_type_id": random.randint(1, 8),
                "discharge_disposition_id": random.randint(1, 30),
                "admission_source_id": random.randint(1, 25),
                "time_in_hospital": random.randint(1, 14),
                "payer_code": random.choice(["?", "CP", "CH", "CM", "SP", "MD", "HM", "UN", "BC", "MC"]),
                "medical_specialty": random.choice(["?", "Emergency/Trauma", "Family/GeneralPractice", 
                                                   "InternalMedicine", "Cardiology", "Surgery-General", 
                                                   "Orthopedics", "Gastroenterology", "Nephrology"]),
                "num_lab_procedures": random.randint(0, 132),
                "num_procedures": random.randint(0, 6),
                "num_medications": random.randint(1, 81),
                "number_outpatient": random.randint(0, 42),
                "number_emergency": random.randint(0, 76),
                "number_inpatient": random.randint(0, 21),
                "diag_1": random.choice(["250.83", "250.00", "414.01", "428.0", "250.40", "?"]),
                "diag_2": random.choice(["250.83", "250.00", "414.01", "428.0", "?", "V45.81"]),
                "diag_3": random.choice(["250.83", "250.00", "?", "V45.81", "401.9"]),
                "number_diagnoses": random.randint(1, 16),
                "max_glu_serum": random.choice(["None", "Norm", ">200", ">300"]),
                "A1Cresult": random.choice(["None", "Norm", ">7", ">8"]),
                "metformin": random.choice(["No", "Up", "Down", "Steady"]),
                "repaglinide": random.choice(["No", "Up", "Down", "Steady"]),
                "nateglinide": random.choice(["No", "Up", "Down", "Steady"]),
                "chlorpropamide": random.choice(["No", "Up", "Down", "Steady"]),
                "glimepiride": random.choice(["No", "Up", "Down", "Steady"]),
                "acetohexamide": random.choice(["No"]),
                "glipizide": random.choice(["No", "Up", "Down", "Steady"]),
                "glyburide": random.choice(["No", "Up", "Down", "Steady"]),
                "tolbutamide": random.choice(["No"]),
                "pioglitazone": random.choice(["No", "Up", "Down", "Steady"]),
                "rosiglitazone": random.choice(["No", "Up", "Down", "Steady"]),
                "acarbose": random.choice(["No", "Up", "Down", "Steady"]),
                "miglitol": random.choice(["No", "Up", "Down", "Steady"]),
                "troglitazone": random.choice(["No"]),
                "tolazamide": random.choice(["No", "Up", "Down", "Steady"]),
                "examide": random.choice(["No"]),
                "citoglipton": random.choice(["No"]),
                "insulin": random.choice(["No", "Up", "Down", "Steady"]),
                "glyburide-metformin": random.choice(["No", "Up", "Down", "Steady"]),
                "glipizide-metformin": random.choice(["No", "Up", "Down", "Steady"]),
                "glimepiride-pioglitazone": random.choice(["No"]),
                "metformin-rosiglitazone": random.choice(["No"]),
                "metformin-pioglitazone": random.choice(["No"]),
                "change": random.choice(["No", "Ch"]),
                "diabetesMed": random.choice(["No", "Yes"])
            }
        }
        
        # Send POST request to /predict endpoint
        with self.client.post(
            "/predict",
            json=patient_data,
            catch_response=True,
            name="predict_readmission"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.failure("API endpoint not found")
            elif response.status_code == 500:
                response.failure("Server error")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def get_health(self):
        """
        Health check task
        Weight: 1 (less frequent than predict)
        """
        self.client.get("/health", name="health_check")
    
    @task(1)
    def get_model_info(self):
        """
        Get model information
        Weight: 1 (less frequent than predict)
        """
        self.client.get("/model-info", name="model_info")

