library(tidyverse)
library(tidymodels)
library(vroom)
# library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
# library(stacks)
library(embed)
install.packages('discrim')
library(discrim)

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
amazon_predictions

amazon_predictions <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

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

# RANDOM FOREST
my_recipe <- recipe(ACTION ~. , data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  # step_dummy(all_nominal_predictors())
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")
  

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,ncol(amazon_train)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

CV_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

rf_predictions <- final_wf %>%
  predict(amazon_test, type = "prob")

rf_predictions <- rf_predictions %>%
  bind_cols(., amazon_test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=rf_predictions, file="rf_predictions.csv", delim=",")

# NAIVE BAYES
my_recipe <- recipe(ACTION ~. , data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  # step_dummy(all_nominal_predictors())
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))

nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

nb_predictions <- final_wf %>%
  predict(amazon_test, type = "prob")

nb_predictions <- nb_predictions %>%
  bind_cols(., amazon_test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=nb_predictions, file="nb_predictions.csv", delim=",")
