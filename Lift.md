# Pratical Machine Learning. Course Project: Writeup
Massimiliano Grassi  

The data for this project come from [this source](http://groupware.les.inf.puc-rio.br/har).

The datasets contain data from accelerometers worn by people that perfored several Dumbbell Bicep Curl exercises in five different ways: one (A) following the correct technique and four (B-E) with incorrect techniques. The training dataset contains data from 19622 observations with the correct classification in the factor variable 'classe'. The testing set contains 20 observation not included in the training dataset that the built algorhitm will have to predict (the correct classification is not provided in the test dataset). 


```r
dir.create("./data")
download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./data/training.csv")

download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="./data/testing.csv")

training<-read.csv("./data/training.csv")
testing<-read.csv("./data/testing.csv")
```

##Model Training

The original training data set was split in a train (80%) and a validation (20%) dataset, in order to use the latter to have an unbiased estimate of the out-of-sample classification performance.


```r
library('caret')

train_validation_split  <- createDataPartition(training$classe, p=0.8, list = FALSE)
train  <- training[train_validation_split,]
validation  <- training[-train_validation_split,]
```

Variable not of interest were removed from the train dataset, e.g. those regarding subjects name and time-stamps or those including mostly missing values. A summary of the removed variables is reported in Appendix 1.


```r
train_cleaned<-train[,-c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
```

All the remaining variables will be employed as cadidate predictors in the process of model building and training. They are all numeric, as indentified by a variable inspection (see Appendix 2). 

Random Forest was chosen as algorithm to build the predicitive model. Training was performed with the train function in the caret package. A k-fold cross-validation strategy (k=20) was be applied during the training in order to choose the combination of tuning parameters that provides the best performance. After that the best tuning parameters combination was identified, the final model was trained with this combination applying the entire train sample.


```r
cross_validation  <- trainControl(method = 'cv', number = 20)

Model_Random_Forest  <- train(classe ~., method = 'rf', data=train_cleaned, trControl = cross_validation)
```


```r
print(Model_Random_Forest)
```

```
## Random Forest 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (20 fold) 
## 
## Summary of sample sizes: 14914, 14914, 14914, 14914, 14915, 14914, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9940123  0.9924255  0.002305245  0.002916205
##   27    0.9936939  0.9920225  0.002391746  0.003025712
##   52    0.9875777  0.9842845  0.004641457  0.005872358
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

#Model Testing

To have an estemation of the out-of-sample classification performance, the fitted model was applied to the validation dataset.


```r
Result_Validation  <- predict(Model_Random_Forest, validation)
confusionmatrixValidation  <- confusionMatrix(Result_Validation, validation$classe)
print(confusionmatrixValidation)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  758    0    0    0
##          C    0    1  684    3    1
##          D    0    0    0  640    0
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9987         
##                  95% CI : (0.997, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9984         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   1.0000   0.9953   0.9986
## Specificity            1.0000   1.0000   0.9985   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   0.9927   1.0000   1.0000
## Neg Pred Value         1.0000   0.9997   1.0000   0.9991   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1744   0.1631   0.1835
## Detection Prevalence   0.2845   0.1932   0.1756   0.1631   0.1835
## Balanced Accuracy      1.0000   0.9993   0.9992   0.9977   0.9993
```

The estimated out-of-sample classification accuracy is 99.8725465% and kappa is 0.9983879, while the estimated classification error is 0.1274535%, as calculated in the validation sample not used to build the madel. 

##Prediction performed with testing dataset

Finally, the model was applied to the testing set and predictions were saved as individual text-files as required for the sumbission to the Coursera web-site.



```r
Result_Testing  <- as.character(predict(Model_Random_Forest, testing))

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

dir.create("./testing_results")
setwd("./testing_results")

pml_write_files(Result_Testing)
```

##Appendix 
##1.

```r
summary(train[,c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)])
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3096   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4912   carlitos:2462   1st Qu.:1.323e+09    1st Qu.:254662      
##  Median : 9814   charles :2849   Median :1.323e+09    Median :496326      
##  Mean   : 9815   eurico  :2499   Mean   :1.323e+09    Mean   :500240      
##  3rd Qu.:14714   jeremy  :2720   3rd Qu.:1.323e+09    3rd Qu.:748393      
##  Max.   :19621   pedro   :2073   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp new_window    num_window    kurtosis_roll_belt
##  05/12/2011 11:24:1204   no :15385   Min.   :  1.0            :15385   
##  28/11/2011 14:14:1204   yes:  314   1st Qu.:224.0   #DIV/0!  :    9   
##  30/11/2011 17:11:1141               Median :425.0   -0.016850:    1   
##  02/12/2011 14:57:1114               Mean   :431.5   -0.021024:    1   
##  05/12/2011 11:25:1107               3rd Qu.:645.0   -0.033935:    1   
##  02/12/2011 13:34:1094               Max.   :864.0   -0.034743:    1   
##  (Other)         :8835                               (Other)  :  301   
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##           :15385            :15385              :15385   
##  #DIV/0!  :   23     #DIV/0!:  314     #DIV/0!  :    8   
##  -0.150950:    3                       0.000000 :    3   
##  -0.684748:    3                       -0.010002:    1   
##  -1.750749:    3                       -0.014020:    1   
##  -1.851307:    3                       -0.015465:    1   
##  (Other)  :  279                       (Other)  :  300   
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt     max_picth_belt 
##           :15385             :15385     Min.   :-94.100   Min.   : 3.00  
##  #DIV/0!  :   23      #DIV/0!:  314     1st Qu.:-88.000   1st Qu.: 5.00  
##  0.000000 :    4                        Median : -5.550   Median :18.00  
##  -3.072669:    3                        Mean   : -5.331   Mean   :12.71  
##  -6.324555:    3                        3rd Qu.: 19.025   3rd Qu.:19.00  
##  -0.189082:    2                        Max.   :180.000   Max.   :28.00  
##  (Other)  :  279                        NA's   :15385     NA's   :15385  
##   max_yaw_belt   min_roll_belt      min_pitch_belt   min_yaw_belt  
##         :15385   Min.   :-180.000   Min.   : 0.00          :15385  
##  -1.1   :   29   1st Qu.: -88.400   1st Qu.: 3.00   -1.1   :   29  
##  -1.4   :   22   Median :  -8.000   Median :16.00   -1.4   :   22  
##  -1.2   :   21   Mean   :  -9.452   Mean   :10.68   -1.2   :   21  
##  -0.9   :   17   3rd Qu.:   8.600   3rd Qu.:17.00   -0.9   :   17  
##  -1.3   :   17   Max.   : 173.000   Max.   :23.00   -1.3   :   17  
##  (Other):  208   NA's   :15385      NA's   :15385   (Other):  208  
##  amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
##  Min.   :  0.000     Min.   : 0.000              :15385     
##  1st Qu.:  0.300     1st Qu.: 1.000       #DIV/0!:    9     
##  Median :  1.000     Median : 1.000       0.00   :    9     
##  Mean   :  4.121     Mean   : 2.035       0.0000 :  296     
##  3rd Qu.:  2.000     3rd Qu.: 2.000                         
##  Max.   :360.000     Max.   :12.000                         
##  NA's   :15385       NA's   :15385                          
##  var_total_accel_belt avg_roll_belt    stddev_roll_belt var_roll_belt    
##  Min.   : 0.000       Min.   :-27.40   Min.   : 0.000   Min.   :  0.000  
##  1st Qu.: 0.100       1st Qu.:  1.10   1st Qu.: 0.168   1st Qu.:  0.000  
##  Median : 0.200       Median :116.35   Median : 0.400   Median :  0.100  
##  Mean   : 0.855       Mean   : 67.53   Mean   : 1.167   Mean   :  6.034  
##  3rd Qu.: 0.300       3rd Qu.:122.97   3rd Qu.: 0.600   3rd Qu.:  0.400  
##  Max.   :16.500       Max.   :157.40   Max.   :10.400   Max.   :108.100  
##  NA's   :15385        NA's   :15385    NA's   :15385    NA's   :15385    
##  avg_pitch_belt    stddev_pitch_belt var_pitch_belt    avg_yaw_belt     
##  Min.   :-51.400   Min.   :0.000     Min.   : 0.000   Min.   :-138.300  
##  1st Qu.:  1.925   1st Qu.:0.200     1st Qu.: 0.000   1st Qu.: -88.200  
##  Median :  5.200   Median :0.400     Median : 0.100   Median :  -6.650  
##  Mean   : -0.162   Mean   :0.558     Mean   : 0.649   Mean   :  -7.771  
##  3rd Qu.: 15.025   3rd Qu.:0.700     3rd Qu.: 0.500   3rd Qu.:  14.125  
##  Max.   : 59.700   Max.   :4.000     Max.   :16.200   Max.   : 173.500  
##  NA's   :15385     NA's   :15385     NA's   :15385    NA's   :15385     
##  stddev_yaw_belt    var_yaw_belt       var_accel_arm      avg_roll_arm    
##  Min.   :  0.000   Min.   :    0.000   Min.   :  0.000   Min.   :-166.67  
##  1st Qu.:  0.100   1st Qu.:    0.010   1st Qu.:  8.787   1st Qu.: -37.35  
##  Median :  0.300   Median :    0.089   Median : 40.294   Median :   0.00  
##  Mean   :  1.491   Mean   :  138.397   Mean   : 51.713   Mean   :  11.42  
##  3rd Qu.:  0.600   3rd Qu.:    0.370   3rd Qu.: 75.012   3rd Qu.:  73.88  
##  Max.   :176.600   Max.   :31183.240   Max.   :331.699   Max.   : 160.78  
##  NA's   :15385     NA's   :15385       NA's   :15385     NA's   :15385    
##  stddev_roll_arm    var_roll_arm       avg_pitch_arm     stddev_pitch_arm
##  Min.   :  0.000   Min.   :    0.000   Min.   :-77.019   Min.   : 0.000  
##  1st Qu.:  1.589   1st Qu.:    2.527   1st Qu.:-20.975   1st Qu.: 1.826  
##  Median :  5.721   Median :   32.730   Median :  0.000   Median : 9.500  
##  Mean   : 10.833   Mean   :  368.817   Mean   : -4.126   Mean   :10.768  
##  3rd Qu.: 14.747   3rd Qu.:  217.506   3rd Qu.:  9.461   3rd Qu.:16.799  
##  Max.   :161.452   Max.   :26066.575   Max.   : 75.659   Max.   :43.412  
##  NA's   :15385     NA's   :15385       NA's   :15385     NA's   :15385   
##  var_pitch_arm       avg_yaw_arm       stddev_yaw_arm   
##  Min.   :   0.000   Min.   :-173.440   Min.   :  0.000  
##  1st Qu.:   3.348   1st Qu.: -32.607   1st Qu.:  2.808  
##  Median :  90.250   Median :   0.000   Median : 16.454  
##  Mean   : 204.192   Mean   :  -0.312   Mean   : 21.763  
##  3rd Qu.: 282.190   3rd Qu.:  33.353   3rd Qu.: 35.551  
##  Max.   :1884.565   Max.   : 152.000   Max.   :163.258  
##  NA's   :15385      NA's   :15385      NA's   :15385    
##   var_yaw_arm        kurtosis_roll_arm kurtosis_picth_arm kurtosis_yaw_arm
##  Min.   :    0.000           :15385            :15385             :15385  
##  1st Qu.:    7.884   #DIV/0! :   58    #DIV/0! :   59     #DIV/0! :    7  
##  Median :  270.751   -0.02438:    1    -0.01311:    1     0.55844 :    2  
##  Mean   :  972.876   -0.04190:    1    -0.02967:    1     0.65132 :    2  
##  3rd Qu.: 1263.882   -0.05695:    1    -0.07394:    1     -0.01548:    1  
##  Max.   :26653.192   -0.08050:    1    -0.10385:    1     -0.01749:    1  
##  NA's   :15385       (Other) :  252    (Other) :  251     (Other) :  301  
##  skewness_roll_arm skewness_pitch_arm skewness_yaw_arm  max_roll_arm   
##          :15385            :15385             :15385   Min.   :-72.30  
##  #DIV/0! :   57    #DIV/0! :   59     #DIV/0! :    7   1st Qu.:  0.00  
##  -0.00051:    1    -0.00184:    1     -1.62032:    2   Median :  5.85  
##  -0.00696:    1    -0.01185:    1     0.55053 :    2   Mean   : 12.45  
##  -0.03359:    1    -0.01247:    1     -0.00311:    1   3rd Qu.: 29.50  
##  -0.04186:    1    -0.02063:    1     -0.00562:    1   Max.   : 85.50  
##  (Other) :  253    (Other) :  251     (Other) :  301   NA's   :15385   
##  max_picth_arm       max_yaw_arm     min_roll_arm    min_pitch_arm    
##  Min.   :-173.000   Min.   : 4.00   Min.   :-89.10   Min.   :-180.00  
##  1st Qu.:  -7.475   1st Qu.:29.00   1st Qu.:-41.90   1st Qu.: -75.90  
##  Median :  14.750   Median :34.00   Median :-22.70   Median : -37.95  
##  Mean   :  32.468   Mean   :34.88   Mean   :-21.08   Mean   : -35.63  
##  3rd Qu.:  95.300   3rd Qu.:40.00   3rd Qu.:  0.00   3rd Qu.:   0.00  
##  Max.   : 180.000   Max.   :62.00   Max.   : 66.40   Max.   : 152.00  
##  NA's   :15385      NA's   :15385   NA's   :15385    NA's   :15385    
##   min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
##  Min.   : 1.0    Min.   :  0.000    Min.   :  0.0       Min.   : 0.00    
##  1st Qu.: 7.0    1st Qu.:  6.862    1st Qu.: 12.1       1st Qu.:13.00    
##  Median :12.0    Median : 30.430    Median : 54.9       Median :22.00    
##  Mean   :14.3    Mean   : 33.526    Mean   : 68.1       Mean   :20.58    
##  3rd Qu.:19.0    3rd Qu.: 51.850    3rd Qu.:114.2       3rd Qu.:28.00    
##  Max.   :38.0    Max.   :119.500    Max.   :359.0       Max.   :51.00    
##  NA's   :15385   NA's   :15385      NA's   :15385       NA's   :15385    
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :15385                 :15385                  :15385        
##  -0.3705:    2          -2.0889:    2           #DIV/0!:  314        
##  -0.5855:    2          #DIV/0!:    2                                
##  -2.0889:    2          -0.0163:    1                                
##  #DIV/0!:    2          -0.0233:    1                                
##  -0.0035:    1          -0.0280:    1                                
##  (Other):  305          (Other):  307                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :15385                 :15385                  :15385        
##  -0.0082:    1          -0.3521:    2           #DIV/0!:  314        
##  -0.0096:    1          -0.7036:    2                                
##  -0.0172:    1          1.0326 :    2                                
##  -0.0224:    1          -0.0053:    1                                
##  -0.0234:    1          -0.0084:    1                                
##  (Other):  309          (Other):  306                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   :-70.00    Min.   :-108.00           :15385    Min.   :-149.60  
##  1st Qu.:-26.88    1st Qu.: -66.28    -0.6   :   18    1st Qu.: -59.85  
##  Median : 15.60    Median :  42.80    -0.3   :   14    Median : -42.60  
##  Mean   : 14.67    Mean   :  33.68    0.0    :   14    Mean   : -41.95  
##  3rd Qu.: 50.80    3rd Qu.: 133.45    0.2    :   14    3rd Qu.: -28.05  
##  Max.   :129.80    Max.   : 155.00    -0.5   :   13    Max.   :  73.20  
##  NA's   :15385     NA's   :15385      (Other):  241    NA's   :15385    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   :-147.00           :15385    Min.   :  0.00         
##  1st Qu.: -92.08    -0.6   :   18    1st Qu.: 16.43         
##  Median : -64.70    -0.3   :   14    Median : 35.83         
##  Mean   : -34.69    0.0    :   14    Mean   : 56.62         
##  3rd Qu.:  14.47    0.2    :   14    3rd Qu.: 88.83         
##  Max.   : 116.60    -0.5   :   13    Max.   :256.48         
##  NA's   :15385      (Other):  241    NA's   :15385          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell var_accel_dumbbell
##  Min.   :  0.00                  :15385          Min.   :  0.000   
##  1st Qu.: 17.35           #DIV/0!:    2          1st Qu.:  0.410   
##  Median : 43.41           0.00   :  312          Median :  1.007   
##  Mean   : 68.36                                  Mean   :  4.701   
##  3rd Qu.:106.11                                  3rd Qu.:  3.476   
##  Max.   :270.84                                  Max.   :230.428   
##  NA's   :15385                                   NA's   :15385     
##  avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell 
##  Min.   :-128.96   Min.   :  0.000      Min.   :    0.00  
##  1st Qu.: -11.67   1st Qu.:  4.776      1st Qu.:   22.81  
##  Median :  51.07   Median : 12.117      Median :  146.82  
##  Mean   :  24.18   Mean   : 21.479      Mean   : 1076.12  
##  3rd Qu.:  64.93   3rd Qu.: 27.014      3rd Qu.:  729.76  
##  Max.   : 125.99   Max.   :123.778      Max.   :15321.01  
##  NA's   :15385     NA's   :15385        NA's   :15385     
##  avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
##  Min.   :-70.73     Min.   : 0.000        Min.   :   0.00   
##  1st Qu.:-40.63     1st Qu.: 3.640        1st Qu.:  13.25   
##  Median :-18.70     Median : 8.313        Median :  69.11   
##  Mean   :-11.61     Mean   :13.566        Mean   : 366.78   
##  3rd Qu.: 11.95     3rd Qu.:19.818        3rd Qu.: 392.75   
##  Max.   : 94.28     Max.   :82.680        Max.   :6836.02   
##  NA's   :15385      NA's   :15385         NA's   :15385     
##  avg_yaw_dumbbell   stddev_yaw_dumbbell var_yaw_dumbbell 
##  Min.   :-117.950   Min.   : 0.000      Min.   :   0.00  
##  1st Qu.: -76.696   1st Qu.: 4.054      1st Qu.:  16.43  
##  Median :   2.044   Median :10.596      Median : 112.28  
##  Mean   :  -0.150   Mean   :16.983      Mean   : 579.63  
##  3rd Qu.:  70.657   3rd Qu.:26.489      3rd Qu.: 701.67  
##  Max.   : 134.905   Max.   :93.652      Max.   :8770.75  
##  NA's   :15385      NA's   :15385       NA's   :15385    
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##         :15385                :15385                 :15385       
##  #DIV/0!:   68         #DIV/0!:   69          #DIV/0!:  314       
##  -0.0227:    1         -0.0073:    1                              
##  -0.0359:    1         -0.0442:    1                              
##  -0.0567:    1         -0.0489:    1                              
##  -0.0781:    1         -0.0523:    1                              
##  (Other):  242         (Other):  241                              
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##         :15385                :15385                 :15385       
##  #DIV/0!:   67         #DIV/0!:   69          #DIV/0!:  314       
##  -0.1912:    2         0.0000 :    3                              
##  -0.0013:    1         -0.0113:    1                              
##  -0.0063:    1         -0.0131:    1                              
##  -0.0088:    1         -0.0405:    1                              
##  (Other):  242         (Other):  239                              
##  max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm 
##  Min.   :-66.60   Min.   :-147.00          :15385   Min.   :-69.400  
##  1st Qu.:  0.00   1st Qu.:   0.00   #DIV/0!:   68   1st Qu.: -5.975  
##  Median : 25.95   Median : 110.00   -1.2   :   25   Median :  0.000  
##  Mean   : 24.43   Mean   :  78.22   -1.3   :   24   Mean   : -0.215  
##  3rd Qu.: 45.75   3rd Qu.: 174.00   -1.0   :   18   3rd Qu.: 11.975  
##  Max.   : 87.90   Max.   : 180.00   -1.5   :   18   Max.   : 47.200  
##  NA's   :15385    NA's   :15385     (Other):  161   NA's   :15385    
##  min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
##  Min.   :-180.00          :15385   Min.   :  0.000       
##  1st Qu.:-175.00   #DIV/0!:   68   1st Qu.:  1.077       
##  Median : -53.00   -1.2   :   25   Median : 18.700       
##  Mean   : -55.55   -1.3   :   24   Mean   : 24.647       
##  3rd Qu.:   0.00   -1.0   :   18   3rd Qu.: 40.208       
##  Max.   : 167.00   -1.5   :   18   Max.   :126.000       
##  NA's   :15385     (Other):  161   NA's   :15385         
##  amplitude_pitch_forearm amplitude_yaw_forearm var_accel_forearm
##  Min.   :  0.0                  :15385         Min.   :  0.00   
##  1st Qu.:  1.2           #DIV/0!:   68         1st Qu.:  6.30   
##  Median : 79.6           0.00   :  246         Median : 22.01   
##  Mean   :133.8                                 Mean   : 33.96   
##  3rd Qu.:350.0                                 3rd Qu.: 53.67   
##  Max.   :359.0                                 Max.   :172.61   
##  NA's   :15385                                 NA's   :15385    
##  avg_roll_forearm   stddev_roll_forearm var_roll_forearm   
##  Min.   :-177.234   Min.   :  0.000     Min.   :    0.000  
##  1st Qu.:  -1.435   1st Qu.:  0.313     1st Qu.:    0.098  
##  Median :   8.502   Median :  8.030     Median :   64.478  
##  Mean   :  32.605   Mean   : 41.929     Mean   : 5287.617  
##  3rd Qu.: 104.004   3rd Qu.: 86.157     3rd Qu.: 7423.031  
##  Max.   : 177.256   Max.   :176.478     Max.   :31144.560  
##  NA's   :15385      NA's   :15385       NA's   :15385      
##  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm 
##  Min.   :-68.17    Min.   : 0.000       Min.   :   0.000  
##  1st Qu.:  0.00    1st Qu.: 0.329       1st Qu.:   0.108  
##  Median : 12.10    Median : 5.574       Median :  31.073  
##  Mean   : 11.73    Mean   : 8.002       Mean   : 140.639  
##  3rd Qu.: 27.79    3rd Qu.:12.697       3rd Qu.: 161.208  
##  Max.   : 62.81    Max.   :47.745       Max.   :2279.617  
##  NA's   :15385     NA's   :15385        NA's   :15385     
##  avg_yaw_forearm   stddev_yaw_forearm var_yaw_forearm   
##  Min.   :-154.79   Min.   :  0.000    Min.   :    0.00  
##  1st Qu.: -26.82   1st Qu.:  0.506    1st Qu.:    0.26  
##  Median :   0.00   Median : 24.194    Median :  585.39  
##  Mean   :  16.14   Mean   : 43.118    Mean   : 4366.16  
##  3rd Qu.:  85.79   3rd Qu.: 75.514    3rd Qu.: 5702.47  
##  Max.   : 169.24   Max.   :197.508    Max.   :39009.33  
##  NA's   :15385     NA's   :15385      NA's   :15385
```

##2.

```r
summary(train_cleaned)
```

```
##    roll_belt         pitch_belt          yaw_belt       total_accel_belt
##  Min.   :-28.900   Min.   :-54.9000   Min.   :-180.00   Min.   : 0.00   
##  1st Qu.:  1.095   1st Qu.:  1.8100   1st Qu.: -88.30   1st Qu.: 3.00   
##  Median :113.000   Median :  5.3000   Median : -13.20   Median :17.00   
##  Mean   : 64.282   Mean   :  0.3761   Mean   : -11.49   Mean   :11.31   
##  3rd Qu.:123.000   3rd Qu.: 14.9000   3rd Qu.:  12.60   3rd Qu.:18.00   
##  Max.   :162.000   Max.   : 60.3000   Max.   : 179.00   Max.   :29.00   
##   gyros_belt_x        gyros_belt_y      gyros_belt_z    
##  Min.   :-1.040000   Min.   :-0.5300   Min.   :-1.4600  
##  1st Qu.:-0.030000   1st Qu.: 0.0000   1st Qu.:-0.2000  
##  Median : 0.030000   Median : 0.0200   Median :-0.1000  
##  Mean   :-0.005602   Mean   : 0.0397   Mean   :-0.1294  
##  3rd Qu.: 0.110000   3rd Qu.: 0.1100   3rd Qu.:-0.0200  
##  Max.   : 2.220000   Max.   : 0.6100   Max.   : 1.6100  
##   accel_belt_x       accel_belt_y     accel_belt_z     magnet_belt_x   
##  Min.   :-120.000   Min.   :-69.00   Min.   :-275.00   Min.   :-52.00  
##  1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:  9.00  
##  Median : -15.000   Median : 34.00   Median :-151.00   Median : 35.00  
##  Mean   :  -5.717   Mean   : 30.11   Mean   : -72.35   Mean   : 55.43  
##  3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  28.00   3rd Qu.: 59.00  
##  Max.   :  85.000   Max.   :164.00   Max.   : 105.00   Max.   :481.00  
##  magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm      
##  Min.   :354.0   Min.   :-621.0   Min.   :-180.00   Min.   :-88.200  
##  1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.80   1st Qu.:-26.000  
##  Median :601.0   Median :-319.0   Median :   0.00   Median :  0.000  
##  Mean   :593.8   Mean   :-345.2   Mean   :  17.68   Mean   : -4.691  
##  3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.20   3rd Qu.: 11.100  
##  Max.   :673.0   Max.   : 293.0   Max.   : 180.00   Max.   : 88.500  
##     yaw_arm          total_accel_arm  gyros_arm_x        gyros_arm_y     
##  Min.   :-180.0000   Min.   : 1.00   Min.   :-6.37000   Min.   :-3.4400  
##  1st Qu.: -42.7000   1st Qu.:17.00   1st Qu.:-1.33000   1st Qu.:-0.8000  
##  Median :   0.0000   Median :27.00   Median : 0.06000   Median :-0.2400  
##  Mean   :  -0.3106   Mean   :25.51   Mean   : 0.04003   Mean   :-0.2567  
##  3rd Qu.:  45.7000   3rd Qu.:33.00   3rd Qu.: 1.57000   3rd Qu.: 0.1600  
##  Max.   : 180.0000   Max.   :66.00   Max.   : 4.87000   Max.   : 2.8400  
##   gyros_arm_z      accel_arm_x       accel_arm_y       accel_arm_z     
##  Min.   :-2.170   Min.   :-404.00   Min.   :-318.00   Min.   :-636.00  
##  1st Qu.:-0.070   1st Qu.:-240.00   1st Qu.: -54.00   1st Qu.:-144.00  
##  Median : 0.230   Median : -41.00   Median :  14.00   Median : -48.00  
##  Mean   : 0.269   Mean   : -59.07   Mean   :  32.31   Mean   : -71.86  
##  3rd Qu.: 0.720   3rd Qu.:  85.00   3rd Qu.: 138.00   3rd Qu.:  23.00  
##  Max.   : 2.990   Max.   : 437.00   Max.   : 308.00   Max.   : 292.00  
##   magnet_arm_x     magnet_arm_y     magnet_arm_z    roll_dumbbell    
##  Min.   :-584.0   Min.   :-392.0   Min.   :-597.0   Min.   :-153.71  
##  1st Qu.:-299.0   1st Qu.: -13.0   1st Qu.: 122.0   1st Qu.: -19.03  
##  Median : 297.0   Median : 199.0   Median : 442.0   Median :  48.00  
##  Mean   : 194.9   Mean   : 154.9   Mean   : 304.3   Mean   :  23.51  
##  3rd Qu.: 640.0   3rd Qu.: 322.0   3rd Qu.: 544.0   3rd Qu.:  67.47  
##  Max.   : 782.0   Max.   : 580.0   Max.   : 694.0   Max.   : 153.38  
##  pitch_dumbbell     yaw_dumbbell      total_accel_dumbbell
##  Min.   :-149.59   Min.   :-148.766   Min.   : 0.00       
##  1st Qu.: -40.94   1st Qu.: -77.592   1st Qu.: 4.00       
##  Median : -21.16   Median :  -4.382   Median :10.00       
##  Mean   : -10.96   Mean   :   1.542   Mean   :13.72       
##  3rd Qu.:  17.33   3rd Qu.:  79.235   3rd Qu.:20.00       
##  Max.   : 137.03   Max.   : 154.952   Max.   :58.00       
##  gyros_dumbbell_x    gyros_dumbbell_y   gyros_dumbbell_z 
##  Min.   :-204.0000   Min.   :-2.10000   Min.   : -2.380  
##  1st Qu.:  -0.0300   1st Qu.:-0.14000   1st Qu.: -0.310  
##  Median :   0.1400   Median : 0.03000   Median : -0.130  
##  Mean   :   0.1604   Mean   : 0.04448   Mean   : -0.125  
##  3rd Qu.:   0.3600   3rd Qu.: 0.21000   3rd Qu.:  0.030  
##  Max.   :   2.2200   Max.   :52.00000   Max.   :317.000  
##  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z  magnet_dumbbell_x
##  Min.   :-419.00   Min.   :-189.00   Min.   :-334.00   Min.   :-643.0   
##  1st Qu.: -51.00   1st Qu.:  -8.00   1st Qu.:-142.00   1st Qu.:-535.0   
##  Median :  -8.00   Median :  40.00   Median :  -1.00   Median :-479.0   
##  Mean   : -28.79   Mean   :  52.52   Mean   : -38.43   Mean   :-328.2   
##  3rd Qu.:  10.00   3rd Qu.: 111.00   3rd Qu.:  38.00   3rd Qu.:-304.0   
##  Max.   : 235.00   Max.   : 315.00   Max.   : 318.00   Max.   : 592.0   
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm     pitch_forearm   
##  Min.   :-3600.0   Min.   :-262.00   Min.   :-180.00   Min.   :-72.50  
##  1st Qu.:  231.0   1st Qu.: -45.00   1st Qu.:  -0.54   1st Qu.:  0.00  
##  Median :  311.0   Median :  13.00   Median :  22.60   Median :  9.16  
##  Mean   :  220.5   Mean   :  45.96   Mean   :  34.21   Mean   : 10.75  
##  3rd Qu.:  391.0   3rd Qu.:  96.00   3rd Qu.: 140.00   3rd Qu.: 28.60  
##  Max.   :  633.0   Max.   : 451.00   Max.   : 180.00   Max.   : 89.80  
##   yaw_forearm      total_accel_forearm gyros_forearm_x   
##  Min.   :-180.00   Min.   :  0.00      Min.   :-22.0000  
##  1st Qu.: -68.30   1st Qu.: 29.00      1st Qu.: -0.2200  
##  Median :   0.00   Median : 36.00      Median :  0.0500  
##  Mean   :  19.53   Mean   : 34.68      Mean   :  0.1559  
##  3rd Qu.: 110.00   3rd Qu.: 41.00      3rd Qu.:  0.5600  
##  Max.   : 180.00   Max.   :108.00      Max.   :  3.9700  
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x   accel_forearm_y 
##  Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00   Min.   :-595.0  
##  1st Qu.: -1.46000   1st Qu.: -0.1800   1st Qu.:-177.50   1st Qu.:  57.0  
##  Median :  0.03000   Median :  0.0800   Median : -57.00   Median : 201.0  
##  Mean   :  0.08165   Mean   :  0.1533   Mean   : -61.49   Mean   : 163.4  
##  3rd Qu.:  1.63000   3rd Qu.:  0.4900   3rd Qu.:  77.00   3rd Qu.: 312.0  
##  Max.   :311.00000   Max.   :231.0000   Max.   : 477.00   Max.   : 923.0  
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-446.00   Min.   :-1280.0   Min.   :-892.0   Min.   :-973    
##  1st Qu.:-181.00   1st Qu.: -617.5   1st Qu.:   5.0   1st Qu.: 191    
##  Median : -40.00   Median : -377.0   Median : 591.0   Median : 513    
##  Mean   : -55.31   Mean   : -313.0   Mean   : 380.3   Mean   : 395    
##  3rd Qu.:  26.00   3rd Qu.:  -77.0   3rd Qu.: 737.0   3rd Qu.: 653    
##  Max.   : 291.00   Max.   :  672.0   Max.   :1480.0   Max.   :1090    
##  classe  
##  A:4464  
##  B:3038  
##  C:2738  
##  D:2573  
##  E:2886  
## 
```

