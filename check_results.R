#install.packages(c("caret", "Hmisc", "boot", "predtools", "moments", "dplyr", "meta", "pROC", "rms", "PRROC"))
# library(caret)
# library(Hmisc)
# library(boot)
# library(predtools)
# library(moments)
# library(dplyr)
# library(meta)
# library(pROC)
# library(rms)
# library(PRROC)
#install.packages("stats")
library(stats)



# Convert probabilities to linear predictors (log-odds)
probs <- c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
lin_predictors <- qlogis(probs)
print(lin_predictors)
data <- data.frame(
  linpreds = lin_predictors,
  outcomes = c(1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0))

# Calibration slope
mtemp <- glm(outcomes ~ linpreds, family="binomial", data=data)
mcslope <- as.numeric(mtemp$coefficients[2])

print(c("Calibration slope: ", mcslope))

# CITL
mtemp <- glm(outcomes ~ offset(linpreds), family="binomial", data=data)
citl <- as.numeric(mtemp$coefficients[1])

print(c("Calibration in the large: ", citl))


oe <- sum(data$outcomes) / sum(probs)
print(c("O/E: ", oe))
