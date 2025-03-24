# Modeling of direct marketing optimization problem

## Description
The objective of this project is to build ML models which can predict the propensity of clients to buy financial products and the likely revenue to be generated. 
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/jaju13/Marketing_Model.git
    ```
2. Navigate into the project folder:
    ```bash
    cd Marketing_Model
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the model, you can use the following commands:
a. To predict Propensity of clients to buy Mutual funds, credit cards and consumer loans, run the command below. 
This command will also generate revenue predicted by the model. 
```bash
python Predict_Propensity_Revenue.py
```
The above command will generate the following files under predicted_results folder :
(i) Propensity_Sale_CC.csv, Propensity_Sale_CL.csv, Propensity_Sale_MF.csv : Contain the predicted propensity to buy MF, CC or CL for all clients wi
(ii) Predicted_Revenue.csv : Contains predicted revenue

b. To predict Top 100 clients , the product they are most likely to buy and the expected revenue to be generated, run the command below:
```bash
python SelectClients.py
```
The above command will generate the following files under predicted_results folder :
(i)Maximum_predicted_revenue_per_client.csv : For all clients without sale information from cleaned data, it calculates the expected revenue and the highest propensity product. 
(ii)Top_100_clients.csv : This file contains top 100 clients who should be targeted with marketing offer. It also lists their expected revenue and the highest propensity product. 

## Results
The results for the exercise are located  'Documents' folder. This contains the following files:
(i) Direct marketing optimization-Executive Summary.pptx : executive summary of the approach and results. 
(ii)Targeted Client List.xlsx

## Folder structure and information
./data : contains pickled data which has been cleaned and outliers have been removed during EDA
./models : Contains models that were selected through grid search , and identified as the best performing models. Identificationw was done through "Modulerized Modeling.iynb" notebook. 
./Documents : Results for the exercise. Details under "Results" section
./Notebooks : This folder contains files used for EDA, Data preparation, Model training and selection. 
./predicted_results : Results from prediction on for clients with no sale information are stored here.

