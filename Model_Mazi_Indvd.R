#############################################################################################################################################################
#
#                                                           LendingClub Default Model
#
#############################################################################################################################################################

wd = "C:/Users/gerard/Documents/LendingClub"
setwd(wd)

library(ggplot2)
library(data.table)
library(caTools)
library(ROCR)
library(caret)
library(pROC)
library(dplyr)
library(bestglm)
library(ggthemes)
library(class)
library(rpart)
library(randomForest)
library(gbm)
library(e1071)
library(naivebayes)

options(scipen=999)

#load("loans.R")

############################################################################################################################################################
# DATA LOAD AND CLEANUP
############################################################################################################################################################

# Source data from:
'https://www.lendingclub.com/info/download-data.action'

# Load data
files = list.files(path = wd, pattern = '*.csv')
loans = do.call(rbind, lapply(files, read.csv, stringsAsFactors = F, skip = 1, header = T, na.strings = c("", NA)))

save(loans, file = 'loans.R')

# Not all fields are available at loan origination, so subset only available fields. Subset further as needed for model development.
var1 = c("loan_status",
              "loan_amnt",
              "term",
              "int_rate",
              "grade",
              "sub_grade",
              "emp_title",
              "installment",
              "emp_length",
              "home_ownership",
              "annual_inc",
              "verification_status",
              "issue_d",
              "purpose",
              "addr_state",
              "dti",
              "delinq_2yrs",
              "earliest_cr_line",
              "fico_range_low",
              "fico_range_high",
              "inq_last_6mths",
              "mths_since_last_delinq",
              "mths_since_last_record",
              "open_acc",
              "pub_rec",
              "revol_bal",
              'revol_util',
              "total_acc",
              "collections_12_mths_ex_med",
              "mths_since_last_major_derog",
              "annual_inc_joint",
              "verification_status_joint",
              "acc_now_delinq",
              "tot_coll_amt",
              "inq_last_12m")
loans = loans[,names(loans) %in% var1]

# Remove loans that fail credit policy because those don't make it into public view
loans = loans[!grepl(pattern = "Does not meet the credit policy", x = loans$loan_status),]
# Remove loans in grace period
loans = loans[loans$loan_status != "In Grace Period",]
# Remove loan_status with NA
loans = loans[!is.na(loans$loan_status),]

# Create a binary variable for performing vs. non-performing
loans$loan_status = 
  factor(
    ifelse(
      loans$loan_status %in% c("Current", "Fully Paid"), 
      FALSE,  
      TRUE
      )
    )

# Set certain variables to factor
#var.factor = c("term","home_ownership","verification_status","purpose","loan_status","addr_state","grade","sub_grade","delinq_2yrs","inq_last_6mths",
#               "mths_since_last_delinq")
#loans[,var.factor] = lapply(loans[,var.factor], factor)

# Set certain variables to decimal
loans$revol_util = as.numeric(sub("%", "", loans$revol_util)) / 100
loans$int_rate = as.numeric(sub("%", "", loans$int_rate)) / 100

# Format Dates
loans$earliest_cr_line = as.Date(paste0("01-", loans$earliest_cr_line), "%d-%b-%Y")

#Factor engineering
loans$credit_years = as.numeric(round((as.Date("2018-09-30") - loans$earliest_cr_line) / 365, 0))
loans$borrowing_need = loans$loan_amnt - loans$revol_bal
loans$borrowing_need = factor(ifelse(loans$borrowing_need < 0, 0, ifelse(loans$borrowing_need > 0 & loans$borrowing_need < 10000, 10000, 20000)))

############################################################################################################################################################
# EXPLORATORY DATA ANALYSIS
############################################################################################################################################################

# Loan Amount (non-monotonic relationship)
summary(loans$loan_amnt)
loans$loan_amnt_bins = cut(loans$loan_amnt, c(quantile(loans$loan_amnt, probs = seq(0, 1, by = 0.1))))
loan_amnt = table(loans$loan_status, loans$loan_amnt_bins)
prop_loan_amnt = round(prop.table(loan_amnt, 2) * 100, 2)
ggplot(melt(prop_loan_amnt), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_loan_amnt") +
  xlab("prop_loan_amnt") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)
ggplot(loans) + geom_histogram(aes(loan_amnt)) + theme_solarized_2(light = FALSE)

# Term
summary(loans$term)
loans$term = factor(loans$term)
term = table(loans$loan_status, loans$term)
prop_term = round(prop.table(term, 2) * 100, 2)
ggplot(melt(prop_term), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_term") +
  xlab("prop_term") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)


# Interest Rate
summary(loans$int_rate)
loans$int_rate_bins = cut(loans$int_rate, c(quantile(loans$int_rate, probs = seq(0, 1, by = 0.1))))
int_rate = table(loans$loan_status, loans$int_rate_bins)
prop_int_rate = round(prop.table(int_rate, 2) * 100, 2)
ggplot(melt(prop_int_rate), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_int_rate") +
  xlab("prop_int_rate") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)
ggplot(loans) + geom_histogram(aes(int_rate)) + theme_solarized_2(light = FALSE)


# Grade
summary(loans$grade)
loans$grade = factor(loans$grade)
grade = table(loans$loan_status, loans$grade)
prop_grade = round(prop.table(grade, 2) * 100, 2)
ggplot(melt(prop_grade), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_grade") +
  xlab("prop_grade") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)


# Sub Grade
summary(loans$sub_grade)
loans$sub_grade = factor(loans$sub_grade)
sub_grade = table(loans$loan_status, loans$sub_grade)
prop_sub_grade = round(prop.table(sub_grade, 2) * 100, 2)
ggplot(melt(prop_sub_grade), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_sub_grade") +
  xlab("prop_sub_grade") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Installment Amount
summary(loans$installment)
loans$installment_bins = cut(loans$installment, c(quantile(loans$installment, probs = seq(0, 1, by = 0.1))))
installment = table(loans$loan_status, loans$installment_bins)
prop_installment = round(prop.table(installment, 2) * 100, 2)
ggplot(melt(prop_installment), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_installment") +
  xlab("prop_installment") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)
ggplot(loans) + geom_histogram(aes(installment)) + theme_solarized_2(light = FALSE)


# Employment Length
summary(loans$emp_length_level)
loans$emp_length_level = factor(ifelse(loans$emp_length == "n/a", "unemployed", "employed"))
emp_length = table(loans$loan_status, loans$emp_length_level)
prop_emp_length = round(prop.table(emp_length, 2) * 100, 2)
ggplot(melt(prop_emp_length), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_emp_length") +
  xlab("prop_emp_length") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)
ggplot(loans) + geom_histogram(aes(emp_length_level)) + theme_solarized_2(light = FALSE)


# Home Ownership
summary(loans$home_ownership)
loans$home_ownership = factor(loans$home_ownership)
home_ownership = table(loans$loan_status, loans$home_ownership)[,c(2,5:6)]
prop_home_ownership = round(prop.table(home_ownership, 2) * 100, 2)
ggplot(melt(prop_home_ownership), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_home_ownership") +
  xlab("prop_home_ownership") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Annual Income
summary(loans$annual_inc)
loans$annual_inc_bins = cut(loans$annual_inc, c(quantile(loans$annual_inc, probs = seq(0, 1, by = 0.05))))
annual_inc = table(loans$loan_status, loans$annual_inc_bins)
prop_annual_inc = round(prop.table(annual_inc, 2) * 100, 2)
ggplot(melt(prop_annual_inc), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_annual_inc") +
  xlab("prop_annual_inc") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)


# Verification Status
summary(loans$verification_status)
loans$verification_status = factor(loans$verification_status)
verification_status = table(loans$loan_status, loans$verification_status)
prop_verification_status = round(prop.table(verification_status, 2) * 100, 2)
ggplot(melt(prop_verification_status), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_verification_status") +
  xlab("prop_verification_status") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Purpose
summary(loans$purpose)
loans$purpose = factor(loans$purpose)
loans$purpose_levels = ifelse(loans$purpose == "credit_card", "credit_card", ifelse(loans$purpose == "debt_consolidation", "debt_consolidation", "other"))
purpose = table(loans$loan_status, loans$purpose_levels)
prop_purpose = round(prop.table(purpose, 2) * 100, 2)
ggplot(melt(prop_purpose), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_purpose") +
  xlab("prop_purpose") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# State
summary(loans$addr_state)
loans$addr_state = factor(loans$addr_state)
addr_state = table(loans$loan_status, loans$addr_state)
prop_adr_state = round(prop.table(addr_state, 2) * 100, 2)
ggplot(melt(prop_adr_state), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_adr_state") +
  xlab("prop_adr_state") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# DTI
summary(loans$dti)
loans$dti_bins = cut(loans$dti, c(quantile(loans$dti, probs = seq(0, 1, by = 0.1), na.rm = TRUE)))
dti = table(loans$loan_status, loans$dti_bins)
prop_dti = round(prop.table(dti, 2) * 100, 2)
ggplot(melt(prop_dti), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_dti") +
  xlab("prop_dti") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Delinquency 2 Year
summary(loans$delinq_2yrs)
loans$delinq_2yrs_levels = factor(loans$delinq_2yrs)
levels(loans$delinq_2yrs_levels) = c("0", "1", "2", rep("3+", 34))
delinq_2yrs = table(loans$loan_status, loans$delinq_2yrs_levels)
prop_delinq_2yrs = round(prop.table(delinq_2yrs, 2) * 100, 2)
ggplot(melt(prop_delinq_2yrs), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_delinq_2yrs") +
  xlab("prop_delinq_2yrs") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Earliest Credit Line
summary(loans$credit_years)
loans$credit_years_bins = cut(loans$credit_years, c(quantile(loans$credit_years, probs = seq(0, 1, by = 0.1))))
credit_years = table(loans$loan_status, loans$credit_years_bins)
prop_credit_years = round(prop.table(credit_years, 2) * 100, 2)
ggplot(melt(prop_credit_years), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_credit_years") +
  xlab("prop_credit_years") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# FICO Score 
summary(loans$fico_range_low)
loans$fico_bins = cut(loans$fico_range_low, c(quantile(loans$fico_range_low, probs = seq(0, 1, by = 0.1))))
fico_range_low = table(loans$loan_status, loans$fico_bins)
prop_fico_range_low = round(prop.table(fico_range_low, 2) * 100, 2)
ggplot(melt(prop_fico_range_low), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_fico_range_low") +
  xlab("prop_fico_range_low") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Inquiries in last 6 months
summary(loans$inq_last_6mths)
loans$inq_last_6mths_levels = factor(loans$inq_last_6mths)
levels(loans$inq_last_6mths_levels) = c("0", "1", "2", "3", rep("4+", 5))
inq_last_6mths = table(loans$loan_status, loans$inq_last_6mths_levels)
prop_inq_last_6mths = round(prop.table(inq_last_6mths, 2) * 100, 2)
ggplot(melt(prop_inq_last_6mths), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("inq_last_6mths") +
  xlab("inq_last_6mths") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Months Since Last Delinquency
summary(loans$mths_since_last_delinq)
loans$mths_since_last_delinq_bins = cut(loans$mths_since_last_delinq, seq(0, 226, by = 12))
mths_since_last_delinq = table(loans$loan_status, loans$mths_since_last_delinq_bins, useNA = 'always')
prop_mths_since_last_delinq = round(prop.table(mths_since_last_delinq, 2) * 100, 2)
ggplot(melt(prop_mths_since_last_delinq), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("mths_since_last_delinq") +
  xlab("mths_since_last_delinq") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)
# Too many NAs. NA doesn't mean no default. remove feature from sample

# Months Since Last Record
summary(loans$mths_since_last_record)
loans$mths_since_last_record_bins = cut(loans$mths_since_last_record, c(seq(0, 130, by = 10)))
mths_since_last_record = table(loans$loan_status, loans$mths_since_last_record_bins)
prop_mths_since_last_record = round(prop.table(mths_since_last_record, 2) * 100, 2)
ggplot(melt(prop_mths_since_last_record), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("mths_since_last_record") +
  xlab("mths_since_last_record") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)
# Too many NAs. NA doesn't mean no default. remove feature from sample

# Number of Open Accounts
summary(loans$open_acc)
loans$open_acc_levels = factor(loans$open_acc)
levels(loans$open_acc_levels) = c(rep("0-5",6),rep("6-10", 5), rep("10+", 79))
open_acc = table(loans$loan_status, loans$open_acc_levels)
prop_open_acc = round(prop.table(open_acc, 2) * 100, 2)
ggplot(melt(prop_open_acc), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_open_acc") +
  xlab("prop_open_acc") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Public Records
summary(loans$pub_rec)
loans$pub_rec_levels = factor(loans$pub_rec)
levels(loans$pub_rec_levels) = c("0", "1", rep("2+", 41))
pub_rec = table(loans$loan_status, loans$pub_rec_levels)
prop_pub_rec = round(prop.table(pub_rec, 2) * 100, 2)
ggplot(melt(prop_pub_rec), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_pub_rec") +
  xlab("prop_pub_rec") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Revolving Balance (non-monotonic relationship)
summary(loans$revol_bal)
loans$revol_bal_bins = cut(loans$revol_bal, c(quantile(loans$revol_bal, probs = seq(0, 1, by = 0.1))))
revol_bal = table(loans$loan_status, loans$revol_bal_bins)
prop_revol_bal = round(prop.table(revol_bal, 2) * 100, 2)
ggplot(melt(prop_revol_bal), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_revol_bal") +
  xlab("prop_revol_bal") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Revolving Utilization
summary(loans$revol_util)
loans$revol_util_bins = cut(loans$revol_util, c(quantile(loans$revol_util, probs = seq(0, 1, by = 0.1), na.rm = TRUE)))
revol_util = table(loans$loan_status, loans$revol_util_bins)
prop_revol_util = round(prop.table(revol_util, 2) * 100, 2)
ggplot(melt(prop_revol_util), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_revol_util") +
  xlab("prop_revol_util") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Total Accounts
summary(loans$total_acc)
loans$total_acc_bins = cut(loans$total_acc, c(quantile(loans$total_acc, probs = seq(0, 1, by = 0.1))))
total_acc = table(loans$loan_status, loans$total_acc_bins)
prop_total_acc = round(prop.table(total_acc, 2) * 100, 2)
ggplot(melt(prop_total_acc), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_total_acc") +
  xlab("prop_total_acc") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Collections last 12 Months
summary(loans$collections_12_mths_ex_med)
loans$collections_12_mths_ex_med_levels = factor(loans$collections_12_mths_ex_med)
levels(loans$collections_12_mths_ex_med_levels) = c("0", rep("1+", 15))
collections_12_mths_ex_med = table(loans$loan_status, loans$collections_12_mths_ex_med_levels)
prop_collections_12_mths_ex_med = round(prop.table(collections_12_mths_ex_med, 2) * 100, 2)
ggplot(melt(prop_collections_12_mths_ex_med), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_collections_12_mths_ex_med") +
  xlab("prop_collections_12_mths_ex_med") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Months Since Last Major Derogatory (do not use, too many NAs, and NA does not mean clean history)
summary(loans$mths_since_last_major_derog)
loans$mths_since_last_major_derog_levels = factor(loans$mths_since_last_major_derog)
levels(loans$mths_since_last_major_derog_levels) = c(rep("1Yr", 11), rep("2Yr", 10), rep("3Yr+", 162))
loans$mths_since_last_major_derog_levels = factor(ifelse(is.na(loans$mths_since_last_major_derog_levels), 
                                                         "no_hist", 
                                                         loans$mths_since_last_major_derog_levels))
mths_since_last_major_derog = table(loans$loan_status, loans$mths_since_last_major_derog_levels)
prop_mths_since_last_major_derog = round(prop.table(mths_since_last_major_derog, 2) * 100, 2)
ggplot(melt(prop_mths_since_last_major_derog), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_mths_since_last_major_derog") +
  xlab("prop_mths_since_last_major_derog") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Joint Annual Income
summary(loans$annual_inc_joint)
# Too many NA's

# Accounts Now Delinquent
summary(loans$acc_now_delinq)
loans$acc_now_delinq_levels = factor(loans$acc_now_delinq)
levels(loans$acc_now_delinq_levels) = c("0", rep("1", 8))
acc_now_delinq = table(loans$loan_status, loans$acc_now_delinq_levels)
prop_acc_now_delinq = round(prop.table(acc_now_delinq, 2) * 100, 2)
ggplot(melt(prop_acc_now_delinq), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_acc_now_delinq") +
  xlab("prop_acc_now_delinq") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Total Collection Amounts ever owed
summary(loans$tot_coll_amt)
loans$tot_coll_amt_levels = factor(loans$tot_coll_amt)
levels(loans$tot_coll_amt_levels) = c("0", rep("1", 15268))
tot_coll_amt = table(loans$loan_status, loans$tot_coll_amt_levels)
prop_tot_coll_amt = round(prop.table(tot_coll_amt, 2) * 100, 2)
ggplot(melt(prop_tot_coll_amt), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_tot_coll_amt") +
  xlab("prop_tot_coll_amt") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

# Inquiries in Last 12 Months
summary(loans$inq_last_12m)
# too many NAs

# Borrowing Need (How much are they borrowing in excess of what they need - the more the worse)
summary(loans$borrowing_need)
borrowing_need = table(loans$loan_status, loans$borrowing_need)
prop_borrowing_need = round(prop.table(borrowing_need, 2) * 100, 2)
ggplot(melt(prop_borrowing_need), aes(Var2, value, fill = Var1)) + 
  geom_bar(position = position_stack(reverse = FALSE), stat = "identity", alpha = 0.7) + 
  ggtitle("prop_borrowing_need") +
  xlab("prop_borrowing_need") +
  ylab("percent of total") + 
  scale_fill_manual(name = "Default", values =  alpha(c("#00BFC4", "#F8766D"), .5)) +
  theme_solarized_2(light = FALSE)

############################################################################################################################################################
# FINAL SAMPLE PREPARATION
############################################################################################################################################################

var2 = c("loan_status","term","grade","emp_length_level","home_ownership","annual_inc","verification_status","fico_range_low","inq_last_6mths_levels",
         "open_acc_levels","revol_util","borrowing_need")
loans2 = loans[,var2]

# Remove home ownership ANY, NONE, OTHER
loans2 = loans2[loans2$home_ownership %in% c("MORTGAGE", "OWN", "RENT"),]

# Remove NAs
loans2 = na.omit(loans2)

# Remove  outliers
loans2 = loans2[loans2$annual_inc < 1000000,]

# Reduce the size of the dataset for faster processing
set.seed(666)
loans2 = 
  loans2[
    sample(
      1:nrow(loans2), 
      nrow(loans2) * 0.10
      )
    , ]

# Split data in Training set and Testing set
set.seed(666)
spl = sample.split(loans2$loan_status, 0.7)
train = subset(loans2, spl == TRUE)
test = subset(loans2, spl == FALSE)

# Balance the data
train = 
  downSample(
    x = train[, !(names(train) %in% c("loan_status"))],
    y = train$loan_status, 
    yname = "loan_status"
    )
prop.table(table(train$loan_status))

rm(loans, loans2)

############################################################################################################################################################
# LOGISTIC REGRESSION
############################################################################################################################################################

# All factors included
glm01 = glm(loan_status ~ ., family = "binomial", data = train)

glm_pred = predict(glm01, test, "response")
glm_roc = roc(test$loan_status, glm_pred)
glm_roc$auc

plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red")
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4))), 
       col = c("red"), lty = 1, cex = 1.0)

# Stepwise selection
glm_null = glm(loan_status ~ 1, family = "binomial", data = train)
glm_full = glm(loan_status ~ ., family = "binomial", data = train)
step_model = step(glm_null, scope = list(lower = glm_null, upper = glm_full), direction = "forward")
summary(step_model)
step_model_pred = predict(step_model, test, "response")
confusionMatrix(table(step_model_pred > 0.15, test$loan_status))
step_model_roc = roc(response = test$loan_status, predictor = step_model_pred)
auc(step_model_roc)
rm(glm_null, glm_full, step_model, step_model_pred, step_model_roc)


############################################################################################################################################################
# CLASSIFICATION TREES
############################################################################################################################################################

# Model
tree01 = rpart(loan_status ~ ., data = train, method = "class", control = rpart.control(cp = 0))
plotcp(tree01)

# Prune
tree01 = 
  prune(
    tree01, 
    cp = 
      tree01$cptable[which.min(tree01$cptable[, "xerror"]), "CP"]
    )

# Plot
plot(tree01, uniform = TRUE, margin = 0.01)
text(tree01)

tree_pred = predict(tree01, test, type = "prob")
tree_roc = roc(test$loan_status, tree_pred[,2])
tree_roc$auc


# Plot ROC
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4))), 
       col = c("red","green"), lty = 1, cex = 1.0)



############################################################################################################################################################
# RANDOM FOREST
############################################################################################################################################################

# Model
rf = randomForest(loan_status ~ ., data = train, ntree = 200)
ggplot(melt(rf$err.rate)) + geom_line(aes(Var1, value, color = Var2), size = 1) + theme_solarized_2(light = FALSE) + xlab("n trees") + ylab("error")

# Tune mtree
rf = tuneRF(train[, !names(train) %in% "loan_status"], train$loan_status, ntreeTry = 200, doBest = TRUE)

# Tune several parameters by grid search
hyper_grid = 
  expand.grid(
    mtry = seq(1, ncol(train)-1, 1), 
    nodesize = seq(1, ncol(train)-1, 1), 
    sampsize = nrow(train) * seq(0.1, 1, 0.1),
    ntree = seq(100, 200, 300)
    )

auc_comp = c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model = 
    randomForest(
      formula = loan_status ~ .,
      data = train,
      #mtry = mtry_iteration[i],
      nodesize = hyper_grid$nodesize[i],
      sampsize = hyper_grid$sampsize[i],
      ntree = hyper_grid$ntree[i]
      )
  
  # Store auc or OOB error for the model
  auc_comp[i] = roc(test$loan_status, predict(model, test, "prob")[,2])$auc 
  #oob_err[i] = model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
#hyper_grid[which.min(oob_err),]
hyper_grid[which.max(auc_comp)]

# Tuned model
rf = randomForest(loan_status ~ ., data = train, ntree = 300, mtry = 2, nodesize = 8, samplesize = 20000)

# Performance
rf_pred = predict(rf, test, "prob")
rf_roc = roc(test$loan_status, rf_pred[,2])
rf_roc$auc

# ROC Plot
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
plot.roc(rf_roc, xlim = c(1, 0), asp = NA, col = "blue", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4)),
                paste0("rf AUC ", round(rf_roc$auc, 4))), 
       col = c("red","green","blue"), lty = 1, cex = 1.0)



############################################################################################################################################################
# BAGGED TREES
############################################################################################################################################################

# Model
bagg = randomForest(loan_status ~ ., data = train, mtry = 11, ntree = 200)
ggplot(melt(bagg$err.rate)) + geom_line(aes(Var1, value, color = Var2), size = 1) + theme_solarized_2(light = FALSE) + xlab("n trees") + ylab("error")

# Tune several parameters by grid search
hyper_grid = 
  expand.grid(
    nodesize = c(2, 5, 8),
    ntree = c(200, 300, 400),
    sampsize = nrow(train) * c(0.35, 0.7, 1)
  )

oob_err = c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model = randomForest(formula = loan_status ~ ., 
                       data = train,
                       mtry = 11,
                       nodesize = hyper_grid$nodesize[i],
                       sampsize = hyper_grid$sampsize[i],
                       ntree = hyper_grid$ntree[i])
  
  # Store OOB error for the model                      
  oob_err[i] = model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
hyper_grid[which.min(oob_err),]

# Tuned model
bagg = randomForest(loan_status ~ ., data = train, mtry = 11, ntree = 300, nodesize = 10, samplesize = 20000)

# Performance
bagg_pred = predict(bagg, test, "prob")
bagg_roc = roc(test$loan_status, bagg_pred[,2])
bagg_roc$auc

# ROC Plot
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
plot.roc(rf_roc, xlim = c(1, 0), asp = NA, col = "blue", add = TRUE)
plot.roc(bagg_roc, xlim = c(1, 0), asp = NA, col = "gold", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4)),
                paste0("rf AUC ", round(rf_roc$auc, 4)),
                paste0("bagg AUC ", round(bagg_roc$auc, 4))), 
       col = c("red","green","blue","gold"), lty = 1, cex = 1.0)





############################################################################################################################################################
# GRADIENT BOOSTING
############################################################################################################################################################
train$loan_status = ifelse(train$loan_status == TRUE, 1, 0)
test$loan_status = ifelse(test$loan_status == TRUE, 1, 0)

gbm = gbm(loan_status ~ ., 
          data = train, 
          distribution = "bernoulli", 
          n.trees = 1000
          #shrinkage=0.1, 
          #interaction.depth=1,
          #bag.fraction = 0.5,
          #n.minobsinnode = 10,
          #cv.folds = 3
)

# Tune several parameters by grid search
hyper_grid = 
  expand.grid(
    shrinkage = seq(0.01, 0.1, 0.01),
    interaction.depth = c(6, 8, 10, 12, 14, 16) 
    #bag.fraction = c(30000, 60000),
    #n.minobsinnode = c(3, 6)
  )

auc_comp = c()

pb = txtProgressBar(min = 0, max = nrow(hyper_grid), initial = 0) 

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model = 
    gbm(
      formula = loan_status ~ .,
      data = train,
      distribution = "bernoulli", 
      n.tree = 695,
      shrinkage = hyper_grid$shrinkage[i],
      interaction.depth = hyper_grid$interaction.depth[i]
      #bag.fraction = hyper_grid$bag.fraction[i],
      #n.minobsinnode = hyper_grid$n.minobsinnode[i]
    )
  
  # Store auc or OOB error for the model
  auc_comp[i] = roc(test$loan_status, predict(model, test, type = "response", n.trees = 695))$auc 
  #oob_err[i] = model$err.rate[nrow(model$err.rate), "OOB"]
  
  setTxtProgressBar(pb,i)
}

# Identify optimal set of hyperparmeters based on OOB error
hyper_grid[which.max(auc_comp)]

gbm = gbm(loan_status ~ ., 
          data = train, 
          distribution = "bernoulli", 
          n.trees = 1000,
          shrinkage=0.03, 
          interaction.depth=10
          #bag.fraction = 0.5,
          #n.minobsinnode = 10,
          #cv.folds = 10
          )
gbm_cv = gbm.perf(gbm, method = "cv")

# Performance
gbm_pred = predict(gbm, test, type = "response", n.trees = 555)
gbm_roc = roc(test$loan_status, gbm_pred)
gbm_roc$auc

# ROC Plot
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
plot.roc(rf_roc, xlim = c(1, 0), asp = NA, col = "blue", add = TRUE)
plot.roc(bagg_roc, xlim = c(1, 0), asp = NA, col = "gold", add = TRUE)
plot.roc(gbm_roc, xlim = c(1, 0), asp = NA, col = "black", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4)),
                paste0("rf AUC ", round(rf_roc$auc, 4)),
                paste0("bagg AUC ", round(bagg_roc$auc, 4)),
                paste0("gbm AUC ", round(gbm_roc$auc, 4))), 
       col = c("red","green","blue","gold","black"), lty = 1, cex = 1.0)


############################################################################################################################################################
# K-NEAREST NEIGHBORS
############################################################################################################################################################

# Normalize training and testing data
normalize = 
  function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }

train_norm = 
  data.frame(
    loan_status = train$loan_status,
    lapply(train[,names(train) %in% c("annual_inc","fico_range_low","revol_util")], normalize),
    lapply(train[,!names(train) %in% c("annual_inc","fico_range_low","revol_util","loan_status")], as.integer)
)

test_norm =
  data.frame(
    loan_status = test$loan_status,
    lapply(test[,names(test) %in% c("annual_inc","fico_range_low","revol_util")], normalize),
    lapply(test[,!names(test) %in% c("annual_inc","fico_range_low","revol_util","loan_status")], as.integer)
  )

train_norm$loan_status = ifelse(train_norm$loan_status == 1, TRUE, FALSE)
test_norm$loan_status = ifelse(test_norm$loan_status == 1, TRUE, FALSE)


set.seed(666)
knn = 
  knn(
    train = train_norm[-1], 
    test = test_norm[-1], 
    cl = train_norm$loan_status, 
    prob = TRUE, 
    k = 10)

knn_pred = attr(knn, "prob")
knn_roc = roc(test_norm$loan_status, knn_pred)
knn_roc$auc

# ROC Plot
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
plot.roc(rf_roc, xlim = c(1, 0), asp = NA, col = "blue", add = TRUE)
plot.roc(bagg_roc, xlim = c(1, 0), asp = NA, col = "gold", add = TRUE)
plot.roc(gbm_roc, xlim = c(1, 0), asp = NA, col = "black", add = TRUE)
plot.roc(knn_roc, xlim = c(1, 0), asp = NA, col = "gray", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4)),
                paste0("rf AUC ", round(rf_roc$auc, 4)),
                paste0("bagg AUC ", round(bagg_roc$auc, 4)),
                paste0("gbm AUC ", round(gbm_roc$auc, 4)),
                paste0("knn AUC ", round(knn_roc$auc, 4))), 
       col = c("red","green","blue","gold","black","gray"), lty = 1, cex = 1.0)


############################################################################################################################################################
# NAIVE BAYES
############################################################################################################################################################

# Binning
train$annual_inc = cut(train$annual_inc, c(quantile(train$annual_inc, probs = seq(0, 1, by = 0.2))))
train$fico_range_low = cut(train$fico_range_low, c(quantile(train$fico_range_low, probs = seq(0, 1, by = 0.2))))
train$revol_util = cut(train$revol_util, c(quantile(train$revol_util, probs = seq(0, 1, by = 0.1))))

test$annual_inc = cut(test$annual_inc, c(quantile(test$annual_inc, probs = seq(0, 1, by = 0.2))))
test$fico_range_low = cut(test$fico_range_low, c(quantile(test$fico_range_low, probs = seq(0, 1, by = 0.2))))
test$revol_util = cut(test$revol_util, c(quantile(test$revol_util, probs = seq(0, 1, by = 0.1))))

# Model
nBayes = naiveBayes(loan_status ~ ., data = train, laplace = 1)

# Performance
nBayes_pred = predict(nBayes, test, type = "raw")
nBayes_roc = roc(test$loan_status, nBayes_pred[,2])
nBayes_roc$auc

# ROC Plot
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
plot.roc(rf_roc, xlim = c(1, 0), asp = NA, col = "blue", add = TRUE)
plot.roc(bagg_roc, xlim = c(1, 0), asp = NA, col = "gold", add = TRUE)
plot.roc(gbm_roc, xlim = c(1, 0), asp = NA, col = "black", add = TRUE)
plot.roc(knn_roc, xlim = c(1, 0), asp = NA, col = "gray", add = TRUE)
plot.roc(nBayes_roc, xlim = c(1, 0), asp = NA, col = "orange", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4)),
                paste0("rf AUC ", round(rf_roc$auc, 4)),
                paste0("bagg AUC ", round(bagg_roc$auc, 4)),
                paste0("gbm AUC ", round(gbm_roc$auc, 4)),
                paste0("knn AUC ", round(knn_roc$auc, 4)),
                paste0("nBayes AUC ", round(nBayes_roc$auc, 4))), 
       col = c("red","green","blue","gold","black","gray","orange"), lty = 1, cex = 1.0)


############################################################################################################################################################
# SUPPORT VECTOR MACHINES
############################################################################################################################################################
'
normalize = 
  function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }

train_norm = 
  data.frame(
    loan_status = train$loan_status,
    lapply(train[,names(train) %in% c("annual_inc","fico_range_low","revol_util")], normalize),
    lapply(train[,!names(train) %in% c("annual_inc","fico_range_low","revol_util","loan_status")], as.integer)
  )

test_norm =
  data.frame(
    loan_status = test$loan_status,
    lapply(test[,names(test) %in% c("annual_inc","fico_range_low","revol_util")], normalize),
    lapply(test[,!names(test) %in% c("annual_inc","fico_range_low","revol_util","loan_status")], as.integer)
  )

train_norm$loan_status = ifelse(train_norm$loan_status == TRUE, 1, -1)
test_norm$loan_status = ifelse(test_norm$loan_status == TRUE, 1, -1)
'
svm = 
  svm(
    loan_status ~ grade + inq_last_6mths_levels, 
    data = train_norm, 
    type = "C-classification", 
    kernel = "radial", 
    probability=TRUE,
    cost = 0.01,
    gamma = 0.5,
    scale = FALSE)

# Performance
svm_pred = predict(svm, test_norm, probability=TRUE)
svm_roc = roc(test_norm$loan_status, attributes(svm_pred)$probabilities[,2])
svm_roc$auc

# Tune Hyperparameters
tune_out = 
  tune.svm(
    x = train_norm[-1], 
    y = as.factor(train_norm$loan_status),
           gamma = 5*10^(-2:2), 
           cost = c(0.01, 0.1, 1, 10, 100), 
           type = "C-classification", 
           kernel = "radial")


#build tuned model
svm = 
  svm(
    loan_status ~ ., 
    data = train_norm, 
    type = "C-classification", 
    kernel = "radial", 
    probability=TRUE,
    cost = 0.01,
    gamma = 0.5,
    scale = FALSE)

# Performance
svm_pred = predict(svm, test_norm, probability=TRUE)
svm_roc = roc(test_norm$loan_status, attributes(svm_pred)$probabilities[,2])
svm_roc$auc

# ROC Plot
plot.roc(glm_roc, xlim = c(1, 0), asp = NA, col = "red", add = FALSE)
plot.roc(tree_roc, xlim = c(1, 0), asp = NA, col = "green", add = TRUE)
plot.roc(rf_roc, xlim = c(1, 0), asp = NA, col = "blue", add = TRUE)
plot.roc(bagg_roc, xlim = c(1, 0), asp = NA, col = "gold", add = TRUE)
plot.roc(gbm_roc, xlim = c(1, 0), asp = NA, col = "black", add = TRUE)
plot.roc(knn_roc, xlim = c(1, 0), asp = NA, col = "gray", add = TRUE)
plot.roc(nBayes_roc, xlim = c(1, 0), asp = NA, col = "orange", add = TRUE)
plot.roc(svm_roc, xlim = c(1, 0), asp = NA, col = "purple", add = TRUE)
legend(x = "bottomright",
       legend=c(paste0("glm AUC ", round(glm_roc$auc, 4)),
                paste0("tree AUC ", round(tree_roc$auc, 4)),
                paste0("rf AUC ", round(rf_roc$auc, 4)),
                paste0("bagg AUC ", round(bagg_roc$auc, 4)),
                paste0("gbm AUC ", round(gbm_roc$auc, 4)),
                paste0("knn AUC ", round(knn_roc$auc, 4)),
                paste0("nBayes AUC ", round(nBayes_roc$auc, 4)),
                paste0("svm AUC ", round(svm_roc$auc, 4))), 
       col = c("red","green","blue","gold","black","gray","orange","purple"), lty = 1, cex = 1.0)
title(main = "Balanced Semi-Optimized")

