# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Standardized Test Analysis

### Overview

**Problem Statement:** 

An increasing number of colleges and universities are adapting to  ACT/SAT-Optional Testing Policies (https://www.fairtest.org/university/optional) , an all time high of more than 1800 schools  or more than 75% of the four-year institutions in the United States - are either test optional or test blind this year (https://www.fairtest.org/sites/default/files/Optional-Growth-Chronology.pdf). 

This project aims to analyse if the new policies have any significant impact in the participation rates, we will be observing over two recent years of data, particularly 2020-2021.

This study is for the College Board, to understand the participation rates of states by region if there is any diminishing trend that they should be concerned of.


### DATASETS

* [`act_2019.csv`](./data/act_2019.csv): 2019 ACT Scores by State ([source](https://blog.prepscholar.com/act-scores-by-state-averages-highs-and-lows))

* [`sat_2019.csv`](./data/sat_2019.csv): 2019 SAT Scores by State ([source](https://blog.prepscholar.com/average-sat-scores-by-state-most-recent))

# Sources/References:
We will be looking at 2019-2021 SAT AND ACT PARTICIPATION by State.
2019 SAT and ACT are provided by GA (link above).
2020 and 2021 are from the website stated below:


* [`act_2020.csv`](./data/act_2020.csv): 2020 ACT Scores by State ([source](https://www.act.org/content/dam/act/unsecured/documents/2020/2020-Average-ACT-Scores-by-State.pdf))

* [`act_2021.csv`](./data/act_2021.csv): 2021 ACT Scores by State ([source](https://www.act.org/content/dam/act/unsecured/documents/2021/2021-Average-ACT-Scores-by-State.pdf))

* [`sat_2020.csv`](./data/sat_2020.csv): 2020 SAT Scores by State ([source](https://research.collegeboard.org/programs/sat/data/2020-sat-suite-annual-report))

* [`sat_2021.csv`](./data/sat_2021.csv): 2021 SAT Scores by State ([source](https://blog.prepscholar.com/average-sat-scores-by-state-most-recent/))

* [`states.csv`](./data/states.csv): State with Code and Region ([source](https://www.kaggle.com/omer2040/usa-states-to-region))


**DATA DICTIONARY** 
## Data Dictionary

SAT:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|year|int|SAT|data from year 2020 to 2021| 
|py_participation_rate|float|SAT|prior year participation rate| 
|py_total|float|SAT|prior year total| 
|code|string|SAT|the state code based on names|
|region|string|SAT|the region of the states|
|rate_diff_from_prior|float|SAT|year's current participation rate minus the prior year|
|percent_range|string|SAT|range from 1-25%, 26-50%, 51-75%, 76-100%|

ACT:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|year|int|ACT|data from year 2020 to 2021| 
|py_participation_rate|float|ACT|prior year participation rate| 
|py_composite|float|ACT|prior year composite| 
|code|string|ACT|the state code based on names|
|region|string|SAT|the region of the states
|rate_diff_from_prior|float|ACT|year's current participation rate minus the prior year|
|percent_range|string|SAT|range from 1-25%, 26-50%, 51-75%, 76-100%|


*Note: Puerto Rico and Virgin Island are excluded during the cleanup process due to missing participation rate from the datasets

**Analysis**

Region Analysis:

    ACT has bigger presence in 3 regions over SAT, it has low presence in Northeast region compared to SAT.
 
In 2021, regions' average participation rate in ACT significantly drop by 9-17%, and 7.5% to 35% for SAT.
West Region having the highest decline in participation rates for ACT, and Northeast for SAT.


Number of states that dropped in participation rate from 2020 and 2021:

   - ACT
       - 2020, 30 states (out of 51 states)
       - 2021, 45 states (out of 51 states)

   - SAT
       - 2020, 20 states (out of 51 states)
       - 2021, 48 states (out of 51 states)
  
  Participation Rate Summary:

   - ACT
       - The overall average participation rate in 2020  is 56.9 %
       - The overall average participation rate in 2021  is 44.55 %

   - SAT
       - The overall average participation rate in 2020 is 48.71 %
       - The overall average participation rate in 2021 is 31.04 %


**Conclusion/Recommendation**

There were  45 to 48 states dropped in participation rates in 2021, with more than 10% decline in average for both SAT and ACT.

The College Board should mitigate further decline as this will have significant impact in future participation rates if not addressed properly. 

They should maintain the strong commitment of states  with high parcipation rates, (e.g. in ACT, 5 states have 100% participation rate in 2020 and 2021 {'Kentucky', 'Mississippi', 'Alabama', 'Tennessee', 'Nevada'}), what is the motivation behind and how this helps colleges and universities and also students to achieve their goals, and probably this can be applied in other states to encourage higher participation rates.  It will also help if College Board will publish more studies and research on how the standard test help to measure  college readiness of each students.

More efforts should focus in regions that have significant decrease (West Region for ACT and Northeast for SAT). We should also look at specific states with -30% decline in participation rates from 2020 to 2021, probably have some  policies to provide more assistance or subsidies to boost  the participation rates.

    - ACT
        - Oklahoma, South (-42 %) 
        - Arizona, West (-36 %)
        - New Mexico, West (-33 %)
        - Minnesota, Midwest (-32 %)
        - Montana, West ( -30%)
        
    
    - SAT
        - Maine,Northeast (-69 %)
        - West Virginia, South ( -53 %)
        - Massachusetts, Northeast  ( -46 %)
        - Colorado, West ( -44 %)
        - California,West ( -43 %)
        - Washington,West ( -42 %)
        - Maryland, South ( -41 %)
        - New Jersey,Northeast ( -34 %)
        - Oregon, West ( -34 %)
        - Michigan, Midwest ( -32 %)
        - Rhode Island,Northeast (-31 %)
        - Connecticut,Northeast (-31 %)







