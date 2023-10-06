library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(embed)

amazon_train <- vroom("./train.csv")
amazon_test <- vroom("./test.csv")

my_recipe <- recipe(ACTION ~. , data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked_train <- bake(prep, new_data = amazon_train)
baked_test <- bake(prep, new_data = amazon_test)


