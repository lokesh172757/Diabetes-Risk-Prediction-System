{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7f405ee1-7485-4dbf-b141-8b26bbc2c947",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: Healthy\n",
      "Probability: 5.04% chance of Diabetes\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# 1. Load the model\n",
    "model = joblib.load('diabetes_model_rf.pkl')\n",
    "# Define column names exactly as they were in your training data\n",
    "cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', \n",
    "        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']\n",
    "\n",
    "# 2. Define a \"Fake Patient\"\n",
    "\n",
    "new_patient = pd.DataFrame([[0, 85, 70, 20, 79, 22.5, 0.2, 25]], columns=cols)\n",
    "\n",
    "# 3. Predict\n",
    "prediction = model.predict(new_patient)\n",
    "probability = model.predict_proba(new_patient)\n",
    "\n",
    "print(f\"Prediction: {'Diabetic' if prediction[0] == 1 else 'Healthy'}\")\n",
    "print(f\"Probability: {probability[0][1]*100:.2f}% chance of Diabetes\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "562360a5-0934-4865-a3a1-37f72db596a0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
