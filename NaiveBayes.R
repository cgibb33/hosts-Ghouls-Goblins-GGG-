library(tidyverse)
library(tidymodels)
library(ggplot2)
library(vroom)
library (embed)

train <- vroom("train.csv.zip")
test <- vroom("test.csv.zip")

ggg_recipe <- recipe(type ~ . , data = train) %>%
  step_mutate(color = as.factor(color))%>%
  step_mutate(id, feature=id) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)


library(discrim)
library(naivebayes)

nbayes_ggg <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification")%>%
  set_engine("naivebayes")


nbayes_ggg_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(nbayes_ggg)


tuning_grid_nbayes <- grid_regular(Laplace(),
                              smoothness(),
                              levels = 3)

## Set up K-fold cv ## Split data for CV
folds_nbayes_ggg <- vfold_cv(train, v = 10, repeats=1)

## Run the Cv
CV_results_nbayes_ggg <- nbayes_ggg_wf %>%
  tune_grid(resamples = folds_nbayes_ggg,
            grid = tuning_grid_nbayes,
            metrics = metric_set(roc_auc))

## Find best tuning parameters
bestTune_nbayes_ggg <- CV_results_nbayes_ggg %>%
  select_best (metric = 'roc_auc')

## Finalize workflow and predict
final_wf_nbayes_ggg <- nbayes_ggg_wf %>%
  finalize_workflow(bestTune_nbayes_ggg) %>%
  fit(data=train)

## Predict
nbayes_ggg_preds <- predict(final_wf_nbayes_ggg, new_data = test, type="class")


## Format the Predictions for Submission to Kaggle 
nbayes_ggg_kaggle_submission <- nbayes_ggg_preds %>%
bind_cols(., test) %>% #Bind predictions with test data 
  select(id, .pred_class) %>% #Just keep datetime and prediction va
  rename(type=.pred_class)
I
## Write out the file
vroom_write(x = nbayes_ggg_kaggle_submission, file="./bayes_ggg.csv", delim = ",")
