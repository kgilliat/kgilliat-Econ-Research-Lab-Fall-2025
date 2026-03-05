import pandas as pd
import numpy as np
import re
from scipy import stats

# Load your data
df = pd.read_csv("data/Utah_full_results_2015-20.csv",
                 on_bad_lines='skip',
                 engine='python')



# Extract year from race_url
def extract_year_from_url(url):
    if pd.isna(url):
        return None
    match = re.search(r'(2015|2016|2017|2018|2019|2020)', str(url))
    if match:
        return int(match.group(1))
    return None

df['year'] = df['race_url'].apply(extract_year_from_url)

# Verify it worked
print(f"Years extracted: {df['year'].value_counts().sort_index()}")
print(f"Rows with missing year: {df['year'].isna().sum()}\n")

# Extract year from race_url
df['year'] = df['race_url'].apply(extract_year_from_url)

# Check for state codes in race_url
#print("Sample URLs:")
#print(df['race_url'].head(10))

# Count URLs by state subdomain
df['state_code'] = df['race_url'].str.extract(r'https://([a-z]{2})\.milesplit\.com')
print("\nURLs by state code:")
print(df['state_code'].value_counts())

# Keep only URLs from ut.milesplit.com
print(f"Before filtering: {len(df)} rows")
df = df[df['race_url'].str.contains(r'https://ut\.milesplit\.com', na=False, case=False)]
print(f"After filtering to UT URLs only: {len(df)} rows")


# Remove entries with missing grade - removes college runners

print(f"Before removing missing grades: {len(df)} rows")
df = df[df['grade'].notna()]
print(f"After removing missing grades: {len(df)} rows")

# Convert finish times to seconds
def time_to_seconds(time_str):
    try:
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except:
        return np.nan

df['finish_seconds'] = df['finish'].apply(time_to_seconds)

####### Identify returners: athletes who appear in multiple years and weren't seniors in first appearance #######


# Identify returners: athletes who appear in multiple years and weren't seniors in first appearance
athlete_years = df.groupby(['athlete', 'team']).agg({
    'year': ['min', 'max', 'count'],
    'grade': 'min'
}).reset_index()

athlete_years.columns = ['athlete', 'team', 'first_year', 'last_year', 'num_years', 'starting_grade']

# Returners are those who appear 2+ years and didn't start as seniors
returners = athlete_years[
    (athlete_years['num_years'] >= 2) & 
    (athlete_years['starting_grade'] < 12)
]

print(f"Total returners (non-seniors): {len(returners)}")

returner_list = returners[['athlete', 'team']].values.tolist()

improvements = []
for athlete, team in returner_list:
    athlete_data = df[(df['athlete'] == athlete) & (df['team'] == team)].sort_values('year')
    if len(athlete_data) >= 2:
        first_time = athlete_data.iloc[0]['finish_seconds']
        last_time = athlete_data.iloc[-1]['finish_seconds']
        if pd.notna(first_time) and pd.notna(last_time):
            improvement = first_time - last_time
            improvements.append(improvement)

print(f"\nReturner Improvement Stats:")
print(f"Average improvement: {np.mean(improvements):.2f} seconds")
print(f"Median improvement: {np.median(improvements):.2f} seconds")
print(f"Percent who improved: {(np.array(improvements) > 0).sum() / len(improvements) * 100:.1f}%")



#### Account for some schools being better than others #####

# Calculate average finish time by team and year
# First, convert finish times to seconds
def time_to_seconds(time_str):
    try:
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except:
        return np.nan

df['finish_seconds'] = df['finish'].apply(time_to_seconds)

# Average performance by team and year
team_performance = df.groupby(['team', 'year']).agg({
    'finish_seconds': ['mean', 'median', 'count'],
    'place': 'mean'
}).reset_index()

team_performance.columns = ['team', 'year', 'avg_time', 'median_time', 'num_runners', 'avg_place']

# Calculate consistency: teams that appear in multiple years with good times
team_consistency = team_performance.groupby('team').agg({
    'avg_time': ['mean', 'std'],
    'year': 'count',
    'num_runners': 'sum'
}).reset_index()

team_consistency.columns = ['team', 'overall_avg_time', 'time_std', 'num_years', 'total_runners']

# Filter for teams with data across multiple years
consistent_teams = team_consistency[team_consistency['num_years'] >= 3].sort_values('overall_avg_time')

print("Top 10 Most Consistent Schools (lowest avg times across years):")
print(consistent_teams.head(10))




###### Freshman Performance #####


# Compare freshman vs other grades
grade_performance = df[df['grade'].isin([9, 10, 11, 12])].groupby('grade').agg({
    'finish_seconds': ['mean', 'median', 'std'],
    'place': 'mean',
    'athlete': 'count'
}).reset_index()

grade_performance.columns = ['grade', 'avg_time', 'median_time', 'time_std', 'avg_place', 'count']

print("\nPerformance by Grade:")
print(grade_performance)

# Statistical test: Are freshman significantly different?
from scipy import stats

freshman_times = df[df['grade'] == 9]['finish_seconds'].dropna()
upperclass_times = df[df['grade'].isin([10, 11, 12])]['finish_seconds'].dropna()

t_stat, p_value = stats.ttest_ind(freshman_times, upperclass_times)
print(f"\nFreshman vs Upperclassmen t-test: t={t_stat:.3f}, p={p_value:.4f}")

# Improvement trajectory: track individual athletes' progression
athlete_progression = df[df['grade'].isin([9, 10, 11, 12])].groupby(['athlete', 'team', 'grade']).agg({
    'finish_seconds': 'mean'
}).reset_index()

# Pivot to see progression across grades
progression_wide = athlete_progression.pivot_table(
    index=['athlete', 'team'],
    columns='grade',
    values='finish_seconds'
)

# Calculate improvement from 9th to 12th grade
progression_wide['improvement_9_to_12'] = progression_wide[9] - progression_wide[12]
progression_wide['pct_improvement'] = (progression_wide['improvement_9_to_12'] / progression_wide[9]) * 100

print("\nAverage improvement from 9th to 12th grade:")
print(f"Time improvement: {progression_wide['improvement_9_to_12'].mean():.2f} seconds")
print(f"Percent improvement: {progression_wide['pct_improvement'].mean():.2f}%")



### Schools with better freshmen ######

# Which schools develop freshman the best?
freshman_by_school = df[df['grade'] == 9].groupby('team').agg({
    'finish_seconds': ['mean', 'count']
}).reset_index()

freshman_by_school.columns = ['team', 'avg_freshman_time', 'num_freshman']

# Filter schools with significant freshman participation
freshman_by_school = freshman_by_school[freshman_by_school['num_freshman'] >= 10]
freshman_by_school = freshman_by_school.sort_values('avg_freshman_time')

print("\nTop 10 Schools with Fastest Freshman:")
print(freshman_by_school.head(10))



