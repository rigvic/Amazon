library(tidyverse)
library(tidymodels)
library(vroom)

amazon_train <- vroom("./train.csv")
amazon_test <- vroom("./test.csv")
