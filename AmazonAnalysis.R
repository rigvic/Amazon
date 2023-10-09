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

amazon_train$ACTION <- as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~. , data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked_train <- bake(prep, new_data = amazon_train)
baked_test <- bake(prep, new_data = amazon_test)

# LOGISTIC REGRESSION
my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazon_train) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data= amazon_test,
                              type = "class") # "class" or "prob" (see doc)

amazon_predictions <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  select(id, .pred_class) %>%
  rename(action = .pred_class)

vroom_write(x=amazon_predictions, file="amazon_predictions.csv", delim=",")





