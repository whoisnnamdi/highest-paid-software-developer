from scipy import stats
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

###############################################################################
# Set random seed given lasso cross validation can generate different results
# each time it's run
###############################################################################
np.random.seed(12345)

###############################################################################
# Read in survey data
###############################################################################
df = pd.read_csv(
    "developer_survey_2018/survey_results_public.csv", low_memory=False)

###############################################################################
# Only consider US respondents
###############################################################################
df = df[df["Country"] == "United States"]

###############################################################################
# Exclude retirees, under 18, and those without any formal education due to 
# small, biased samples
###############################################################################
df = df[df["Employment"] != "Retired"]
df = df[df["Age"] != "Under 18 years old"]
df = df[df["FormalEducation"] != "I never completed any formal education"]

###############################################################################
# Drop unnecessary columns. Rename ConvertedSalary to Income
###############################################################################
df = df.drop(["Country", "Salary", "Currency", "CurrencySymbol"], axis = 1)
df = df.rename(columns = {"ConvertedSalary": "Income"})
df = df.drop([x for x in df.columns if "AssessJob" in x], axis = 1)
df = df.drop([x for x in df.columns if "AssessBenefits" in x], axis = 1)
df = df.drop([x for x in df.columns if "JobContactPriorities" in x], axis = 1)
df = df.drop([x for x in df.columns if "JobEmailPriorities" in x], axis = 1)
df = df.drop([x for x in df.columns if "AdBlocker" in x], axis = 1)
df = df.drop([x for x in df.columns if "AdsAgreeDisagree" in x], axis = 1)
df = df.drop([x for x in df.columns if "AdsActions" in x], axis = 1)
df = df.drop([x for x in df.columns if "AdsPriorities" in x], axis = 1)
df = df.drop([x for x in df.columns if "StackOverflow" in x], axis = 1)
df = df.drop([x for x in df.columns if "HypotheticalTools" in x], axis = 1)
df = df.drop([x for x in df.columns if "TimeAfterBootcamp" in x], axis = 1)
df = df.drop([x for x in df.columns if "Satisfaction" in x], axis = 1)
df = df.drop([x for x in df.columns if "AIFuture" in x], axis = 1)
df = df.drop([x for x in df.columns if "AIInteresting" in x], axis = 1)
df = df.drop([x for x in df.columns if "AIResponsible" in x], axis = 1)
df = df.drop([x for x in df.columns if "AIDangerous" in x], axis = 1)
df = df.drop([x for x in df.columns if "NextYear" in x], axis = 1)
df = df.drop([x for x in df.columns if "Ethic" in x], axis = 1)
df = df.drop([x for x in df.columns if "Survey" in x], axis = 1)
df = df.drop([x for x in df.columns if "AgreeDisagree" in x], axis = 1)
df = df.drop([x for x in df.columns if "UpdateCV" in x], axis = 1)

###############################################################################
# Drop any respondents who didn't provide income
# Drop very low or very high income
# Log transform income
###############################################################################
df = df.dropna(subset = ["Income"])
df = df[(df["Income"] > 10000) & (df["Income"] <= 250000)]
df["Income"] = np.log(df["Income"])

df = df.fillna("no_answer")

###############################################################################
# Exclude respondents who selected multiple gender, race, or sexual orientation
# options
###############################################################################
df = df[~df["Gender"].str.contains(";")]
df = df[~df["RaceEthnicity"].str.contains(";")]
df = df[~df["SexualOrientation"].str.contains(";")]

###############################################################################
# Drop respondent column, which we don't need, and reset index
###############################################################################
df = df.drop("Respondent", axis = 1)
df = df.reset_index(drop=True)

###############################################################################
# Create list of controls
###############################################################################
controls_list = list(df.columns)
controls_list.remove("Income")

###############################################################################
# Setup omitted category for each control dummy ("no" if control has no missing
# values, otherwise "no_answer"). Then fill in "no_answer" for non-answers
###############################################################################
omitted = {}

###############################################################################
# Text clean up function
###############################################################################
def text_clean (text):
    text = str(text).replace(" ", "_").replace("-", "_").replace(
        ",", "_").replace(".", "").replace("+", "p").replace("#", "s").replace(
            "/", "_").replace("'", "").replace("ʼ", "").replace(
                "(", "_").replace(")", "_").replace("’", "").replace(
                    "__", "_").replace("__", "_").replace("“", "").replace(
                        "”", "").replace(":", "_").replace("&", "_").lower()

    return text

###############################################################################
# Setup omitted category to be the most common response
# For questions where multiple answere were possible, selecting single highly
# popular answer as omitted category
# For a few of the controls, manually setting omitted category to be something
# more intuitive, like the "lowest" possible answer
###############################################################################
for c in controls_list:
    omitted[c] = text_clean(df[c].value_counts().idxmax())

omitted["LanguageWorkedWith"] = "no_answer"
omitted["FrameworkWorkedWith"] = "no_answer"
omitted["DatabaseWorkedWith"] = "no_answer"
omitted["PlatformWorkedWith"] = "no_answer"
omitted["IDE"] = "no_answer"
omitted["VersionControl"] = "no_answer"
omitted["DevType"] = "no_answer"
omitted["Methodology"] = "no_answer"
omitted["CommunicationTools"] = "no_answer"
omitted["Gender"] = "male"
omitted["RaceEthnicity"] = "white_or_of_european_descent"
omitted["SexualOrientation"] = "straight_or_heterosexual"
omitted["EducationTypes"] = "no_answer"
omitted["SelfTaughtTypes"] = "no_answer"
omitted["YearsCoding"] = "0_2_years"
omitted["YearsCodingProf"] = "0_2_years"
omitted["YearsCodingProf"] = "0_2_years"
omitted["Age"] = "18_24_years_old"
omitted["CompanySize"] = "fewer_than_10_employees"
omitted["NumberMonitors"] = "1"
omitted["WakeTime"] = "before_5_00_am"
omitted["HoursOutside"] = "less_than_30_minutes"
omitted["HoursComputer"] = "9_12_hours"

###############################################################################
# Create control dummies, drop original raw controls
###############################################################################
controls = {}
vec = CountVectorizer(token_pattern = r"(?u)\b\w+\b")

for c in controls_list:
    df.loc[:,c] = df[c].apply(text_clean)
    vec.fit(df[c].values)
    controls[c] = [k for k in vec.vocabulary_]

    for i in controls[c]:
        df[c+"_"+i] = df[c].apply(lambda x: i in str(x).split(";")) * 1

    df = df.drop(c+"_"+omitted[c], axis = 1)
    df = df.drop(c, axis = 1)

###############################################################################
# Double Selection
###############################################################################
results = {}
results["coef"] = {}
results["std_error"] = {}

for k,v in controls.items():
            for i in v:
                    if i != omitted[k]:
                        X, y = df.drop("Income", axis=1).astype(float).copy(), df["Income"].copy()

                        t = k + "_" + i

                        T = X[t]
                        X = X.drop(t, axis = 1)
                        clf = LassoCV(cv = 5, max_iter = 10000, selection = "random", n_jobs = -1)

                        sfm = SelectFromModel(clf)

                        H, K = sfm.fit(X, y).get_support(), sfm.fit(X, T).get_support()

                        U = H | K

                        X_U = X.loc[:, U].copy()

                        X_U.loc[:, t] = T.copy()

                        X_U2 = sm.add_constant(X_U)
                        est = sm.OLS(endog=y, exog=X_U2).fit()
                        results["coef"][t] = est.params[-1]
                        results["std_error"][t] = est.bse[-1]

                        print(t + ": Done")

results_df = pd.DataFrame(data = list(zip([v for k,v in results["coef"].items()], [v for k,v in results["std_error"].items()], (df.sum()/len(df)).drop("Income"))), columns = ["coef", "std_error", "percentage"])

results_df["index"] = [k for k,v in results["coef"].items()]
results_df = results_df.set_index("index", drop = True)
results_df.rename_axis(None, inplace=True)

results_df.to_csv("results.csv")