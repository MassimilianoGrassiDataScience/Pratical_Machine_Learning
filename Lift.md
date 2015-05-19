# Dumbbell Bicep Curl
Massimiliano Grassi  
19 maggio 2015  

#Pratical Machine Learning. Course Project: Writeup

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

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
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
```

The estimated out-of-sample classification accuracy is 99.4646954% and kappa is 0.9932287, while the estimated classification error is 0.5353046%, as calculated in the validation sample not used to build the madel. 

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
##  Min.   :    1   adelmo  :3095   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4912   carlitos:2492   1st Qu.:1.323e+09    1st Qu.:254720      
##  Median : 9812   charles :2826   Median :1.323e+09    Median :496867      
##  Mean   : 9810   eurico  :2453   Mean   :1.323e+09    Mean   :501290      
##  3rd Qu.:14718   jeremy  :2740   3rd Qu.:1.323e+09    3rd Qu.:752294      
##  Max.   :19622   pedro   :2093   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp new_window    num_window  kurtosis_roll_belt
##  28/11/2011 14:14:1197   no :15374   Min.   :  1            :15374   
##  05/12/2011 11:24:1194   yes:  325   1st Qu.:221   #DIV/0!  :    8   
##  30/11/2011 17:11:1185               Median :424   -1.908453:    2   
##  05/12/2011 11:25:1147               Mean   :430   -0.016850:    1   
##  02/12/2011 14:58:1099               3rd Qu.:643   -0.021024:    1   
##  05/12/2011 14:23:1097               Max.   :864   -0.025513:    1   
##  (Other)         :8780                             (Other)  :  312   
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##           :15374            :15374              :15374   
##  #DIV/0!  :   28     #DIV/0!:  325     #DIV/0!  :    7   
##  -0.150950:    3                       0.000000 :    3   
##  -0.684748:    3                       0.422463 :    2   
##  -1.851307:    3                       -0.003095:    1   
##  1.326417 :    3                       -0.014020:    1   
##  (Other)  :  285                       (Other)  :  311   
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt     max_picth_belt 
##           :15374             :15374     Min.   :-94.300   Min.   : 3.0   
##  #DIV/0!  :   28      #DIV/0!:  325     1st Qu.:-88.000   1st Qu.: 5.0   
##  0.000000 :    4                        Median : -5.100   Median :18.0   
##  -2.156553:    3                        Mean   : -8.646   Mean   :12.8   
##  -0.475418:    2                        3rd Qu.: 15.700   3rd Qu.:19.0   
##  -0.587156:    2                        Max.   :180.000   Max.   :30.0   
##  (Other)  :  286                        NA's   :15374     NA's   :15374  
##   max_yaw_belt   min_roll_belt     min_pitch_belt   min_yaw_belt  
##         :15374   Min.   :-180.00   Min.   : 0.00          :15374  
##  -1.4   :   26   1st Qu.: -88.40   1st Qu.: 3.00   -1.4   :   26  
##  -1.1   :   24   Median :  -8.00   Median :16.00   -1.1   :   24  
##  -1.2   :   22   Mean   : -11.74   Mean   :10.63   -1.2   :   22  
##  -0.9   :   20   3rd Qu.:   9.70   3rd Qu.:17.00   -0.9   :   20  
##  -1.3   :   17   Max.   : 173.00   Max.   :23.00   -1.3   :   17  
##  (Other):  216   NA's   :15374     NA's   :15374   (Other):  216  
##  amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
##  Min.   :  0.000     Min.   : 0.000              :15374     
##  1st Qu.:  0.300     1st Qu.: 1.000       #DIV/0!:    8     
##  Median :  1.000     Median : 1.000       0.00   :   10     
##  Mean   :  3.092     Mean   : 2.172       0.0000 :  307     
##  3rd Qu.:  2.120     3rd Qu.: 2.000                         
##  Max.   :360.000     Max.   :12.000                         
##  NA's   :15374       NA's   :15374                          
##  var_total_accel_belt avg_roll_belt    stddev_roll_belt var_roll_belt    
##  Min.   : 0.000       Min.   :-27.40   Min.   : 0.000   Min.   :  0.000  
##  1st Qu.: 0.100       1st Qu.:  1.10   1st Qu.: 0.200   1st Qu.:  0.000  
##  Median : 0.200       Median :116.30   Median : 0.400   Median :  0.100  
##  Mean   : 0.883       Mean   : 67.04   Mean   : 1.313   Mean   :  7.222  
##  3rd Qu.: 0.300       3rd Qu.:123.70   3rd Qu.: 0.700   3rd Qu.:  0.500  
##  Max.   :16.500       Max.   :157.40   Max.   :14.200   Max.   :200.700  
##  NA's   :15374        NA's   :15374    NA's   :15374    NA's   :15374    
##  avg_pitch_belt    stddev_pitch_belt var_pitch_belt    avg_yaw_belt    
##  Min.   :-51.400   Min.   :0.000     Min.   : 0.000   Min.   :-138.30  
##  1st Qu.:  2.200   1st Qu.:0.200     1st Qu.: 0.000   1st Qu.: -88.20  
##  Median :  5.300   Median :0.400     Median : 0.100   Median :  -6.60  
##  Mean   :  0.975   Mean   :0.607     Mean   : 0.763   Mean   : -10.63  
##  3rd Qu.: 15.700   3rd Qu.:0.700     3rd Qu.: 0.500   3rd Qu.:  13.80  
##  Max.   : 59.700   Max.   :4.000     Max.   :16.200   Max.   : 173.40  
##  NA's   :15374     NA's   :15374     NA's   :15374    NA's   :15374    
##  stddev_yaw_belt    var_yaw_belt      var_accel_arm     avg_roll_arm    
##  Min.   :  0.000   Min.   :    0.00   Min.   :  0.00   Min.   :-166.67  
##  1st Qu.:  0.100   1st Qu.:    0.01   1st Qu.: 11.47   1st Qu.: -37.58  
##  Median :  0.300   Median :    0.08   Median : 41.23   Median :   0.00  
##  Mean   :  0.972   Mean   :   37.91   Mean   : 55.03   Mean   :  14.22  
##  3rd Qu.:  0.700   3rd Qu.:    0.51   3rd Qu.: 77.95   3rd Qu.:  76.35  
##  Max.   :109.200   Max.   :11928.47   Max.   :331.70   Max.   : 163.33  
##  NA's   :15374     NA's   :15374      NA's   :15374    NA's   :15374    
##  stddev_roll_arm    var_roll_arm       avg_pitch_arm     stddev_pitch_arm
##  Min.   :  0.000   Min.   :    0.000   Min.   :-81.773   Min.   : 0.000  
##  1st Qu.:  1.082   1st Qu.:    1.171   1st Qu.:-24.184   1st Qu.: 1.465  
##  Median :  5.737   Median :   32.912   Median :  0.000   Median : 7.934  
##  Mean   : 10.974   Mean   :  368.024   Mean   : -5.410   Mean   : 9.999  
##  3rd Qu.: 14.915   3rd Qu.:  222.469   3rd Qu.:  7.303   3rd Qu.:15.899  
##  Max.   :161.452   Max.   :26066.575   Max.   : 70.703   Max.   :43.412  
##  NA's   :15374     NA's   :15374       NA's   :15374     NA's   :15374   
##  var_pitch_arm       avg_yaw_arm       stddev_yaw_arm   
##  Min.   :   0.000   Min.   :-173.440   Min.   :  0.000  
##  1st Qu.:   2.147   1st Qu.: -23.199   1st Qu.:  1.874  
##  Median :  62.952   Median :   0.000   Median : 16.856  
##  Mean   : 181.672   Mean   :   5.243   Mean   : 22.121  
##  3rd Qu.: 252.786   3rd Qu.:  39.083   3rd Qu.: 35.910  
##  Max.   :1884.565   Max.   : 152.000   Max.   :177.044  
##  NA's   :15374      NA's   :15374      NA's   :15374    
##   var_yaw_arm        kurtosis_roll_arm kurtosis_picth_arm kurtosis_yaw_arm
##  Min.   :    0.000           :15374            :15374             :15374  
##  1st Qu.:    3.511   #DIV/0! :   65    #DIV/0! :   67     #DIV/0! :   11  
##  Median :  284.121   -0.04190:    1    -0.00484:    1     0.55844 :    2  
##  Mean   : 1031.646   -0.05051:    1    -0.01311:    1     0.65132 :    2  
##  3rd Qu.: 1289.565   -0.05695:    1    -0.02967:    1     -0.01548:    1  
##  Max.   :31344.568   -0.08050:    1    -0.07394:    1     -0.01749:    1  
##  NA's   :15374       (Other) :  256    (Other) :  254     (Other) :  308  
##  skewness_roll_arm skewness_pitch_arm skewness_yaw_arm  max_roll_arm   
##          :15374            :15374             :15374   Min.   :-73.10  
##  #DIV/0! :   64    #DIV/0! :   67     #DIV/0! :   11   1st Qu.: -2.30  
##  -0.00051:    1    -0.01185:    1     -1.62032:    2   Median :  3.70  
##  -0.00696:    1    -0.01247:    1     0.55053 :    2   Mean   : 10.46  
##  -0.01884:    1    -0.02063:    1     -0.00311:    1   3rd Qu.: 24.70  
##  -0.03359:    1    -0.02652:    1     -0.00562:    1   Max.   : 85.50  
##  (Other) :  257    (Other) :  254     (Other) :  308   NA's   :15374   
##  max_picth_arm      max_yaw_arm     min_roll_arm    min_pitch_arm    
##  Min.   :-173.00   Min.   : 4.00   Min.   :-89.10   Min.   :-180.00  
##  1st Qu.:   0.00   1st Qu.:30.00   1st Qu.:-41.90   1st Qu.: -69.80  
##  Median :  32.60   Median :35.00   Median :-22.40   Median : -31.60  
##  Mean   :  37.64   Mean   :35.92   Mean   :-20.91   Mean   : -31.06  
##  3rd Qu.:  96.00   3rd Qu.:42.00   3rd Qu.:  0.00   3rd Qu.:   0.00  
##  Max.   : 180.00   Max.   :65.00   Max.   : 63.50   Max.   : 152.00  
##  NA's   :15374     NA's   :15374   NA's   :15374    NA's   :15374    
##   min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
##  Min.   : 1.0    Min.   :  0.00     Min.   :  0.0       Min.   : 0.00    
##  1st Qu.: 8.0    1st Qu.:  5.40     1st Qu.:  9.8       1st Qu.:13.00    
##  Median :13.0    Median : 28.00     Median : 55.6       Median :22.00    
##  Mean   :14.8    Mean   : 31.36     Mean   : 68.7       Mean   :21.12    
##  3rd Qu.:19.0    3rd Qu.: 50.15     3rd Qu.:114.8       3rd Qu.:29.00    
##  Max.   :38.0    Max.   :119.50     Max.   :360.0       Max.   :51.00    
##  NA's   :15374   NA's   :15374      NA's   :15374       NA's   :15374    
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :15374                 :15374                  :15374        
##  #DIV/0!:    3          -0.5464:    2           #DIV/0!:  325        
##  -0.2583:    2          -2.0833:    2                                
##  -2.0889:    2          -2.0889:    2                                
##  -0.0035:    1          -0.0233:    1                                
##  -0.0073:    1          -0.0280:    1                                
##  (Other):  316          (Other):  317                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :15374                 :15374                  :15374        
##  #DIV/0!:    2          -0.2328:    2           #DIV/0!:  325        
##  -0.0096:    1          -0.7036:    2                                
##  -0.0172:    1          0.1090 :    2                                
##  -0.0224:    1          1.0326 :    2                                
##  -0.0234:    1          -0.0053:    1                                
##  (Other):  319          (Other):  316                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   :-70.10    Min.   :-112.90           :15374    Min.   :-134.90  
##  1st Qu.:-27.90    1st Qu.: -66.80    -0.6   :   17    1st Qu.: -60.70  
##  Median : 12.30    Median :  43.60    -0.3   :   15    Median : -45.10  
##  Mean   : 13.03    Mean   :  33.29    0.2    :   14    Mean   : -42.75  
##  3rd Qu.: 50.50    3rd Qu.: 134.30    -0.5   :   13    3rd Qu.: -27.70  
##  Max.   :129.80    Max.   : 154.50    -0.7   :   13    Max.   :  73.20  
##  NA's   :15374     NA's   :15374      (Other):  253    NA's   :15374    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   :-146.20           :15374    Min.   :  0.00         
##  1st Qu.: -92.00    -0.6   :   17    1st Qu.: 13.80         
##  Median : -66.90    -0.3   :   15    Median : 35.80         
##  Mean   : -34.42    0.2    :   14    Mean   : 55.78         
##  3rd Qu.:  16.70    -0.5   :   13    3rd Qu.: 86.06         
##  Max.   : 120.90    -0.7   :   13    Max.   :256.48         
##  NA's   :15374      (Other):  253    NA's   :15374          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell var_accel_dumbbell
##  Min.   :  0.00                  :15374          Min.   : 0.000    
##  1st Qu.: 16.83           #DIV/0!:    3          1st Qu.: 0.362    
##  Median : 42.26           0.00   :  322          Median : 0.995    
##  Mean   : 67.71                                  Mean   : 4.025    
##  3rd Qu.:104.51                                  3rd Qu.: 3.195    
##  Max.   :270.84                                  Max.   :64.175    
##  NA's   :15374                                   NA's   :15374     
##  avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell 
##  Min.   :-128.96   Min.   :  0.000      Min.   :    0.00  
##  1st Qu.: -11.88   1st Qu.:  4.749      1st Qu.:   22.55  
##  Median :  47.80   Median : 12.225      Median :  149.44  
##  Mean   :  22.43   Mean   : 21.252      Mean   : 1069.67  
##  3rd Qu.:  62.73   3rd Qu.: 26.685      3rd Qu.:  712.09  
##  Max.   : 125.99   Max.   :123.778      Max.   :15321.01  
##  NA's   :15374     NA's   :15374        NA's   :15374     
##  avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
##  Min.   :-70.73     Min.   : 0.000        Min.   :   0.00   
##  1st Qu.:-42.41     1st Qu.: 3.414        1st Qu.:  11.66   
##  Median :-20.80     Median : 8.384        Median :  70.29   
##  Mean   :-13.23     Mean   :13.225        Mean   : 352.19   
##  3rd Qu.: 12.59     3rd Qu.:19.315        3rd Qu.: 373.08   
##  Max.   : 93.93     Max.   :82.680        Max.   :6836.02   
##  NA's   :15374      NA's   :15374         NA's   :15374     
##  avg_yaw_dumbbell   stddev_yaw_dumbbell var_yaw_dumbbell  
##  Min.   :-117.950   Min.   :  0.00      Min.   :    0.00  
##  1st Qu.: -76.976   1st Qu.:  3.96      1st Qu.:   15.68  
##  Median :  -6.654   Median : 10.14      Median :  102.91  
##  Mean   :  -0.053   Mean   : 16.89      Mean   :  596.66  
##  3rd Qu.:  71.269   3rd Qu.: 26.19      3rd Qu.:  685.87  
##  Max.   : 134.905   Max.   :107.09      Max.   :11467.91  
##  NA's   :15374      NA's   :15374       NA's   :15374     
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##         :15374                :15374                 :15374       
##  #DIV/0!:   65         #DIV/0!:   65          #DIV/0!:  325       
##  -0.8079:    2         -0.0073:    1                              
##  -0.9169:    2         -0.0442:    1                              
##  -0.0227:    1         -0.0523:    1                              
##  -0.0567:    1         -0.0891:    1                              
##  (Other):  254         (Other):  256                              
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##         :15374                :15374                 :15374       
##  #DIV/0!:   64         #DIV/0!:   65          #DIV/0!:  325       
##  -0.4126:    2         0.0000 :    4                              
##  -0.0004:    1         -0.6992:    2                              
##  -0.0013:    1         -0.0113:    1                              
##  -0.0063:    1         -0.0131:    1                              
##  (Other):  256         (Other):  252                              
##  max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm 
##  Min.   :-66.60   Min.   :-151.00          :15374   Min.   :-69.400  
##  1st Qu.:  0.00   1st Qu.:   0.00   #DIV/0!:   65   1st Qu.: -6.500  
##  Median : 27.10   Median : 118.00   -1.3   :   24   Median :  0.000  
##  Mean   : 24.67   Mean   :  84.46   -1.2   :   21   Mean   : -0.761  
##  3rd Qu.: 47.20   3rd Qu.: 175.00   -1.4   :   20   3rd Qu.: 12.600  
##  Max.   : 87.90   Max.   : 180.00   -1.6   :   20   Max.   : 56.500  
##  NA's   :15374    NA's   :15374     (Other):  175   NA's   :15374    
##  min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
##  Min.   :-180.00          :15374   Min.   :  0.00        
##  1st Qu.:-176.00   #DIV/0!:   65   1st Qu.:  1.50        
##  Median : -66.30   -1.3   :   24   Median : 19.10        
##  Mean   : -60.07   -1.2   :   21   Mean   : 25.43        
##  3rd Qu.:   0.00   -1.4   :   20   3rd Qu.: 40.21        
##  Max.   : 167.00   -1.6   :   20   Max.   :126.00        
##  NA's   :15374     (Other):  175   NA's   :15374         
##  amplitude_pitch_forearm amplitude_yaw_forearm var_accel_forearm
##  Min.   :  0.0                  :15374         Min.   :  0.000  
##  1st Qu.:  2.0           #DIV/0!:   65         1st Qu.:  7.035  
##  Median : 87.9           0.00   :  260         Median : 23.340  
##  Mean   :144.5                                 Mean   : 34.925  
##  3rd Qu.:351.0                                 3rd Qu.: 54.053  
##  Max.   :360.0                                 Max.   :172.606  
##  NA's   :15374                                 NA's   :15374    
##  avg_roll_forearm  stddev_roll_forearm var_roll_forearm  
##  Min.   :-177.23   Min.   :  0.000     Min.   :    0.00  
##  1st Qu.:   0.00   1st Qu.:  0.505     1st Qu.:    0.25  
##  Median :  15.28   Median :  8.783     Median :   77.14  
##  Mean   :  35.68   Mean   : 43.501     Mean   : 5459.43  
##  3rd Qu.: 112.89   3rd Qu.: 92.335     3rd Qu.: 8525.67  
##  Max.   : 177.26   Max.   :179.171     Max.   :32102.24  
##  NA's   :15374     NA's   :15374       NA's   :15374     
##  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm 
##  Min.   :-68.17    Min.   : 0.000       Min.   :   0.000  
##  1st Qu.:  0.00    1st Qu.: 0.446       1st Qu.:   0.199  
##  Median : 11.99    Median : 5.955       Median :  35.458  
##  Mean   : 11.64    Mean   : 8.287       Mean   : 149.750  
##  3rd Qu.: 28.71    3rd Qu.:13.175       3rd Qu.: 173.579  
##  Max.   : 68.51    Max.   :47.745       Max.   :2279.617  
##  NA's   :15374     NA's   :15374        NA's   :15374     
##  avg_yaw_forearm   stddev_yaw_forearm var_yaw_forearm   
##  Min.   :-155.06   Min.   :  0.000    Min.   :    0.00  
##  1st Qu.: -26.87   1st Qu.:  0.693    1st Qu.:    0.48  
##  Median :   0.00   Median : 26.281    Median :  690.68  
##  Mean   :  17.84   Mean   : 46.601    Mean   : 4897.62  
##  3rd Qu.:  85.97   3rd Qu.: 93.860    3rd Qu.: 8809.77  
##  Max.   : 169.24   Max.   :197.508    Max.   :39009.33  
##  NA's   :15374     NA's   :15374      NA's   :15374
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

