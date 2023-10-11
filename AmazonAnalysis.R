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
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  # step_dummy(all_nominal_predictors())
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))

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
                              type = "prob") # "class" or "prob" (see doc)

amazon_predictions <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  select(id, .pred_class) %>%
  rename(action = .pred_class)

vroom_write(x=amazon_predictions, file="amazon_predictions.csv", delim=",")


# PENALIZED LOGISTIC REGRESSION
my_recipe <- recipe(ACTION ~. , data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  # step_dummy(all_nominal_predictors())
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))

pen_mod <- logistic_reg(mixture = tune() , penalty = tune()) %>%
  set_engine("glmnet")

pen_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

CV_results <- pen_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <-
  pen_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

pen_predictions <- final_wf %>%
  predict(amazon_test, type = "prob")

pen_predictions <- pen_predictions %>%
  bind_cols(., amazon_test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=pen_predictions, file="pen_predictions.csv", delim=",")

bestTune
