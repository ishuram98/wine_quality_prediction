# MIS 720 - E-Business and Big Data Infrastructures Final Project
# Team Name : Avalanche Analytics
# Wine Quality Prediction
# Ishwarya Ramkumar, Poojitha Pakeeru, Sushma Bulusu

install.packages('caret', dependencies = TRUE)

rm(list=ls())
#Loading libraries
library(caret)
library(party)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(e1071)
library(corrplot)
library(dplyr)
library(pROC)
library(class)


#Loading dataset
wine = read.csv("winequalityN.csv", header = TRUE)

#dimensions
dim(wine)

#Data types 
str(wine)

head(wine)
summary(wine)


# dummy-coding categorical variable - type is the only categorical variable with possible values: "white" or "red"
# Convert categorical variable into dummy variables.
wine.dmodel <- dummyVars( ~ ., data=wine, fullRank=T)

wine <- as.data.frame(predict(wine.dmodel, wine))

# renaming dummy variable typewhite as type
colnames(wine)[colnames(wine) == "typewhite"] <- "type"


# white wine is assigned 1 and red wine is assigned 0 dummy-coding type variable
table(wine$type)

# All missing values are very low percentage in numerical features. We will fill the null values with the mean for each attribute.
# checking for the null values
colSums(is.na(wine))

# fill null values with the mean of each column, except for the 'type' column
for (col in names(wine)) {
  if (col != "type") {
    wine[[col]] <- ifelse(is.na(wine[[col]]), mean(wine[[col]], na.rm = TRUE), wine[[col]])
  }
}

# re-check for null values
colSums(is.na(wine))


# Box plot - to visualize outliers and remove them

numeric_data = wine[,c("fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol","quality")]
boxplot(x = as.list(numeric_data))

# Calculate the first and third quartiles and interquartile range (IQR) for free sulplhur dioxide
Q1 <- quantile(wine$free.sulfur.dioxide, 0.25)
Q3 <- quantile(wine$free.sulfur.dioxide, 0.75)
IQR <- Q3 - Q1

# Calculate lower and upper limits for outliers for free sulplhur dioxide
lower_limit <- Q1 - 1.5 * IQR
upper_limit <- Q3 + 1.5 * IQR
cat("Lower limit:", lower_limit, "\n")
cat("Upper limit:", upper_limit, "\n")

# Subset the data to remove outliers based on the calculated limits for free sulphur dioxide
df2 <- wine[wine$free.sulfur.dioxide > lower_limit & wine$free.sulfur.dioxide < upper_limit, ]
cat("Number of outliers removed:", nrow(wine) - nrow(df2), "\n")

# Calculate the first and third quartiles and interquartile range (IQR) for total sulphur dioxide
Q1 <- quantile(df2$total.sulfur.dioxide, 0.25)
Q3 <- quantile(df2$total.sulfur.dioxide, 0.75)
IQR <- Q3 - Q1

# Calculate lower and upper limits for outliers for total sulphur dioxide
lower_limit <- Q1 - 1.5 * IQR
upper_limit <- Q3 + 1.5 * IQR
cat("Lower limit:", lower_limit, "\n")
cat("Upper limit:", upper_limit, "\n")

# Subset the data to remove outliers based on the calculated limits for total sulphur dioxide
df3 <- df2[df2$total.sulfur.dioxide > lower_limit & df2$total.sulfur.dioxide < upper_limit, ]
cat("Number of outliers removed:", nrow(df2) - nrow(df3), "\n")


# Calculate the first and third quartiles and interquartile range (IQR) for residual sugar
Q1 <- quantile(df3$residual.sugar, 0.25)
Q3 <- quantile(df3$residual.sugar, 0.75)
IQR <- Q3 - Q1

# Calculate lower and upper limits for outliers for residual sugar
lower_limit <- Q1 - 1.5 * IQR
upper_limit <- Q3 + 1.5 * IQR
cat("Lower limit:", lower_limit, "\n")
cat("Upper limit:", upper_limit, "\n")

# Subset the data to remove outliers based on the calculated limits for residual sugar
df4 <- df3[df3$residual.sugar > lower_limit & df3$residual.sugar < upper_limit, ]
cat("Number of outliers removed:", nrow(df3) - nrow(df4), "\n")

wine <- df4

boxplot(x = as.list(df4[,-1]))

nrow(wine)
head(wine)

# remove highly correlated variables
#We have set the cutoff value as 0.7, so if there is correlation above 0.7 between two different parameters, 
#the correlation is high enough, so we can remove one of the features. Hence, we decided to remove the feature 'total sulfur dioxide'.
corr_matrix <- cor(numeric_data)
cor(numeric_data)
high_correlation <- findCorrelation(corr_matrix, cutoff = 0.7, verbose = FALSE)
high_correlation 
# as total sulphur dioxide is highly co-related, we are dropping it
# total sulphur dioxide is 8th column in wine dataframe, so dropping the 8th column
wine <- wine[, -8]

head(wine)

#histogram
# numeric_data with total sulphur dioxide column removed
numeric_data = wine[,c("fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","density","pH","sulphates","alcohol","quality")]

#  most of the features are normally distributed.
par(mfrow=c(2,6)) 
for (i in 1:ncol(numeric_data)) {
  hist(numeric_data[,i], main = colnames(numeric_data)[i], xlab = colnames(numeric_data)[i], ylab = "Frequency")
}


cor(wine[, 2:length(wine)])


#Data types
str(wine)

wine = na.omit(wine)
head(wine)

dim(wine)

dev.off()

corrplot(cor(wine))

# Plot the distribution of quality variable 

ggplot(wine, aes(x = factor(quality))) +
  geom_bar() +
  scale_fill_manual(values = c("#3366CC", "#DC3912")) +
  labs(title = "Quality Distribution") +
  xlab("Quality") +
  ylab("Count") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# As we can observe, the most common wine quality is 6, when 3 is the lowest wine quality, and 9 is the highest wine quality.
table(wine$quality)

# splitting quality into 2 classes - quality equal to and above 6 are considered elite(good quality)
# and quality below 6 are considered non-elite(bad quality)
wine$quality <-  ifelse(wine$quality >= 6, 1, 0)


head(wine)

# converting predictor variable to factor for classification
wine$quality = as.factor(wine$quality)

table(wine$quality)

str(wine)

n= dim(wine)[1]
n



# spilitting data into training and testing data in the ratio 70 : 30 
# seed is set for reproducability
set.seed(100)
train.index = sample(n, round(0.7*n))
train = wine[train.index, ]
dim(train)
test = wine[-train.index, ]
dim(test)

colnames(wine)

test.y= test$quality
test.y

# Logistic regression
set.seed(100)
glm.fit = glm(quality~type+ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + density + pH + sulphates + alcohol,family=binomial, data=train)
summary(glm.fit)

# training - confusion matrix and accuracy

log_train <- predict(glm.fit, newdata = train, type = "response")

log_train.categorical <- ifelse(log_train > 0.5,"1","0")
log_train.categorical <- as.factor(log_train.categorical)
cm_log_train<-confusionMatrix(data=log_train.categorical,reference=train$quality)
cm_log_train


# testing - confusion matrix and accuracy

log_test <- predict(glm.fit, newdata = test, type = "response")

log_test.categorical <- ifelse(log_test > 0.5,"1","0")
log_test.categorical<-as.factor(log_test.categorical)

cm_log_test<-confusionMatrix(data=log_test.categorical,reference=test$quality)
cm_log_test


# Logistic ROC curve
roc(test$quality, log_test, plot = TRUE, legacy.axes = TRUE,main= "ROC of Logistic Regression",
    col="#377eb8", lwd =1, print.auc = TRUE, asp=1)






## Naive bayes

set.seed(100)
ctrl <- trainControl(method = "cv", number=10, summaryFunction=twoClassSummary,
                     classProbs=T, savePredictions=T)

naive_bayes <- train(quality ~ ., 
                     data = train,
                     method = "naive_bayes",
                     usepoisson = TRUE,
                     na.action = na.pass)

# Viewing the model
naive_bayes

# training data - confusion matrix and accuracy
nb_train <- predict(naive_bayes, newdata = train,type= "raw")
nb_train.categorical <- as.factor(nb_train)
cm_nb_train<-confusionMatrix(data=nb_train.categorical,reference=train$quality)
cm_nb_train


# testing data - confusion matrix and accuracy
nb_test <- predict(naive_bayes, newdata = test, type = "raw")
nb_test.categorical<-as.factor(nb_test)

cm_nb_test<-confusionMatrix(data=nb_test.categorical,reference=test$quality)
cm_nb_test


# Naive Bayes ROC curve
nb_test_roc <- predict(naive_bayes, test, type = "prob")

roc(test$quality, nb_test_roc[ ,2], plot = TRUE, legacy.axes = TRUE,
    main= "ROC of Naive Bayes",col="#377eb8", lwd =1, print.auc = TRUE,asp=1)






## KNN

set.seed(100)
ctrl_knn <- trainControl(method = "cv",
                         summaryFunction = defaultSummary,
                         number =5)

set.seed(100)
knn <- train(quality ~ ., 
             data = train,
             method = "knn",
             metric = "Accuracy",
             tuneLength = 2,
             na.action = na.pass,
             trControl=ctrl_knn,
)


# training data - confusion matrix and accuracy
knn_train <- predict(knn, newdata = train, type = "raw")
knn_train.categorical <- as.factor(knn_train)
cm_knn_train<-confusionMatrix(data=knn_train.categorical,reference=train$quality)
cm_knn_train


# testing data - confusion matrix and accuracy
knn_test <- predict(knn, newdata = test, type = "raw")
knn_test.categorical<-as.factor(knn_test)
cm_knn_test<-confusionMatrix(data=knn_test.categorical,reference=test$quality)
cm_knn_test


# KNN ROC Curve

knn_test_roc <- predict(knn, test, type = "prob")

roc(test$quality, knn_test_roc[ ,2], plot = TRUE,lwd = 1,legacy.axes = TRUE, main= "ROC of KNN",
    col="#377eb8",print.auc = TRUE , asp =1)



## Decision Tree
set.seed(100)
ctrl_decision_tree <- trainControl(method="repeatedcv",repeats = 3)
decision_tree <- train(quality ~., 
                     data = train, 
                     method = "rpart",
                     parms = list(split = "information"),
                     trControl=ctrl_decision_tree,
                     tuneLength = 10,
                     na.action = na.pass)

#model summary
decision_tree 

# plotting the tree
prp(decision_tree$finalModel, box.palette = "Blue", tweak = 1.2)

# observing the best performing variables in the model
plot(varImp(decision_tree))

# training data -confusion matrix and accuracy
dt_train <- predict(decision_tree, newdata = train, type = "raw")
dt_train.categorical <- as.factor(dt_train)

cm_dt_train<-confusionMatrix(data=dt_train.categorical,reference=train$quality)
cm_dt_train


# testing data - confusion matrix and accuracy
dt_test <- predict(decision_tree, newdata = test, type = "raw")
dt_test.categorical<-as.factor(dt_test)

cm_dt_test<-confusionMatrix(data=dt_test.categorical,reference=test$quality)
cm_dt_test


# Decision Tree ROC curve
dt_test_roc <- predict(decision_tree, test, type = "prob")

roc(test$quality, dt_test_roc[ ,2], plot = TRUE, legacy.axes = TRUE,
    main= "ROC of Decision Tree",col="#377eb8", lwd =1, print.auc = TRUE, asp =1)

# Decision tree has the highest accuracy of 75.57% among all the other models

