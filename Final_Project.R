title: "Practical Machine Learning Project"
author: "Alfred Homere Ngandam Mfomdoum"
date: "September 4, 2016"
he goal of this project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. we may use any of the other variables to predict with. we should create a report describing how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices. We will also use our prediction model to predict 20 different test cases.

# Getting the Data
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Loading the data from the directory ie to memory solely
getwd()
setwd("Coursera")
list.dirs()
setwd("./Practical_Machine_Learning/Week4/Project")
list.files()
DaSetTraining <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
DaSetTraining
DaSetTesting <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
DaSetTesting

# Partitioning the training set into two
library("caret")
inTrain <- createDataPartition(y=DaSetTraining$classe, p=0.6, list=FALSE)
myTraining <- DaSetTraining[inTrain, ]; myTesting <- DaSetTraining[-inTrain, ]
dim(myTraining); dim(myTesting)
# Cleaning the data
myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)
# Running the code to create another subset without NZV variables
myNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
                                      "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
                                      "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
                                      "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
                                      "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
                                      "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
                                      "max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
                                      "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
                                      "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
                                      "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
                                      "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
                                      "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
                                      "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
                                      "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
                                      "stddev_yaw_forearm", "var_yaw_forearm")
myTraining <- myTraining[!myNZVvars]
#To check the new N?? of observations
dim(myTraining)

# Transformation 2: Killing first column of Dataset - ID Removing first ID variable so that it does not interfer with ML Algorithm
myTraining <- myTraining[c(-1)]
myTraining

# Cleaning Variables with too many NAs. For Variables that have more than a 60% threshold of NA’s I’m going to leave them out

trainingV3 <- myTraining #creating another subset to iterate in loop
for(i in 1:length(myTraining)) { #for every column in the training dataset
  if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .6 ) { #if n?? NAs > 60% of total observations
    for(j in 1:length(trainingV3)) {
      if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:
        trainingV3 <- trainingV3[ , -j] #Remove that column
      }   
    } 
  }
}
#To check the new N?? of observations
dim(trainingV3)

#Seting back to our set:
myTraining <- trainingV3
rm(trainingV3)
# Now let us do the exact same 3 transformations but for our myTesting and testing data sets.
clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58]) #already with classe column removed
myTesting <- myTesting[clean1]
DaSetTesting <- DaSetTesting[clean2]

#To check the new N?? of observations
dim(myTesting)
#To check the new N?? of observations
dim(DaSetTesting)

#Note: The last column - problem_id - which is not equal to training sets, was also "automagically" removed
#No need for this code:
#testing <- testing[-length(testing)]

# In order to ensure proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.

for (i in 1:length(DaSetTesting) ) {
  for(j in 1:length(myTraining)) {
    if( length( grep(names(myTraining[i]), names(DaSetTesting)[j]) ) ==1)  {
      class(DaSetTesting[j]) <- class(myTraining[i])
    }      
  }      
}
#And to make sure Coertion really worked, simple smart ass technique:
DaSetTesting <- rbind(myTraining[2, -58] , DaSetTesting) #note row 2 does not mean anything, this will be removed right.. now:
DaSetTesting <- DaSetTesting[-1,]

install.packages("rpart")
library("rpart")

# Using ML algorithms for prediction: Decision Tree
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
#to view the decision tree with fancy run this command:
install.packages("rpart.plot")
library("rpart.plot")
install.packages("rattle")
library("rattle")
fancyRpartPlot(modFitA1)
# Predicting
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
# (Moment of truth) Using confusion Matrix to test results:
confusionMatrix(predictionsA1, myTesting$classe)

#Overall Statistics

#               Accuracy : 0.8728          
#                 95% CI : (0.8652, 0.8801)
#    No Information Rate : 0.2845          
#    P-Value [Acc > NIR] : < 2.2e-16       

#                  Kappa : 0.839

# Using ML algorithms for prediction: Random Forests
install.packages("randomForest")
library("randomForest")
modFitB1 <- randomForest(classe ~. , data=myTraining)
modFitB1

# Predicting in-sample error:
  
  predictionsB1 <- predict(modFitB1, myTesting, type = "class")
  
# (Moment of truth) Using confusion Matrix to test results:
  
  confusionMatrix(predictionsB1, myTesting$classe)
  
  # Overall Statistics
  
 #  Accuracy : 0.9992          
  # 95% CI : (0.9983, 0.9997)
  # No Information Rate : 0.2845          
  # P-Value [Acc > NIR] : < 2.2e-16       
  
  # Kappa : 0.999           
  # Mcnemar's Test P-Value : NA 
# Random Forests yielded better Results, as expected!
  
  
  
  # Generating Files to submit as answers for the Assignment:
  # Finally, using the provided Test Set out-of-sample error.
  
  # For Random Forests we use the following formula, which yielded a much better prediction in in-sample:
  
predictionsB2 <- predict(modFitB1, DaSetTesting, type = "class")
  
# Function to generate files with predictions to submit for assignment

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)
