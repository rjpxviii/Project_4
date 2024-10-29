# Violent Crimes in Texas

## Overview
This repository contains a project aimed at predicting the safety of certain cities in Texas on a per capita basis given various factors. After the initial analysis of data, the goal changed to predicting crime rate per 1000 people. 

## Table of Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Key Aspects to Explore](#key-aspects-to-explore)
- [Databases to be Used](#databases-to-be-used)
- [Machine Learning](#machine-learning)
- [Ethical  Considerations](#ethical-considerations)
- [Breakdown of Tasks](#breakdown-of-tasks)
- [Data Standardization and Analysis](#data-standardization-and-analysis)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)




## Project Description
For this project, we will analyze Census data alongside the National Incident-Based Reporting System to assess factors related to violent crime in Texas. 
Our goal is to predict the safety of specific cities on a per capita basis. We will employ the classification model Random Forest, and will categorize the data based on various features. Throughout the project, we will leverage Python, Pandas, Matplotlib, and Tableau for different analytical tasks.
After analyzing the data, our goal changed to predicting the crime rate per 1000 people.


## Key Aspects to Explore

1. **Predict Crime Rate:**

    - Predict crime rate based on city-based feature, like poverty rate.
    - Use the RandomForest model to predict the crime rate.

2. **Visualize Crime:**

     - Creating an interactive map that shows:
         * The cities in Texas where the color and size of the markers is determined by the crime rate. 

## Databases to be Used
* [US Census Bureau:](https://api.census.gov/data/2022/acs/acs5) A Dataset that will allow us to get valuable information on the following features:
    - Median Household Income
    - Poverty Rates
        - Percentage of Males under the Poverty Line
        - Percentage of Females under the Poverty Line
        - Percentage of adults over 25 under the Poverty Line
        - Percentage of Poverty to City Population
    - Education Rates
        - Percentage of adults with High School or Less Education
        - Percentage of adults with High School Degree or Equivalent
        - Percentage of adults with Some college and/or Associates
        - Percentage of adults with Bachelor’s Degree or more
    - Population
    - Housing Data
        - Percent of Owner Occupied housing
        - Percent of Renter Occupied housing
        - Median Rent
    - Total Offenses of Crime

* [FBI Crime Data Explorer 2022 for Texas](https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads) An Exel file that displayed the number of crimes in Texas.

## Machine Learning

**Model Used**

The model we used for this project was RandomForest

**Target Variable**

The target variable was crime rate per 1000. 

Note: Standard notation was used by Researchers to determine the rate of crime.

**Optimization**

1. Removed extraneous column of index 
    - The left column on the original
2. Removed the feature "Percentage of Owner Lived" as it was deemed irrelevant and had a stronger than expected pull to the data.
3. Utilized a new model to help identify the best fit of important features using GridSearchCV


## Ethical Considerations 
When conducting this project on housing trends in Central Texas, we made conscious efforts to adhere to ethical considerations, particularly in how we obtained and handled data. All data was sourced from publicly available and reputable sources, such as the U.S. Census Bureau. We ensured that no personal or sensitive information was collected. Additionally, we carefully followed data usage guidelines specified by the data providers to maintain integrity and trustworthiness in our analysis. By adhering to these ethical practices, we aimed to provide insights that are both accurate and responsible, while respecting the rights of individuals and communities represented in the data.

## Breakdown of Tasks

* **Michael Sanchez:** Data collection and preprocessing (Census data), and PPT presentation.
* **Griselda Rodriguez:** Data collection and preprocessing (FBI Crime Data), README.md file, and PPT presentation.
* **Adil Baksh:** Data visualization, and PPT presentation.
* **Santiago Cardenas:** Machine Learning Code, and PPT presentation.

## Data Standardization and Analysis

* **Data Standardization:** Multiple API calls were made in order to obtain the necessary data for this project 
* **Merged datasets:** Created a new data frame merging the census data and the crime data

## Visualizations
### Crime Rate in Texas
https://public.tableau.com/views/TexasCrimes/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
<img width="746" alt="Screenshot 2024-10-28 at 7 24 47 PM" src="https://github.com/user-attachments/assets/8a7373fe-e5b4-41bf-bb3f-4412ea1fc456">

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## References

1. Safewise "100 Safeset Cities in the US"
   https://www.safewise.com/safest-cities-america/
2. GridSearchCV Documentation
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
