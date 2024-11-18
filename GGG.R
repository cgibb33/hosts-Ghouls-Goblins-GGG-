library(vroom)
library(tidymodels)
library(keras)
library(tensorflow)

#Read In data
test <- vroom("test.csv.zip")
misstrain <- vroom("trainWithMissingValues.csv")
train <- vroom("train.csv.zip")

recipe <- recipe(id ~ .,data = misstrain)%>%
  step_impute_mean(all_numeric_predictors())%>%
  step_impute_mode(all_nominal_predictors())

prep <- prep(recipe)
baked <- bake(prep, misstrain)

rmse_vec(train[is.na(misstrain)],
         baked[is.na(misstrain)])




### NN

# Neural Network Recipe
nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%  # Turn color to factor and dummy encode
  step_range(all_numeric_predictors(), min = 0, max = 1)  # Scale to [0,1]

# Prepare the recipe
nnprep <- prep(nn_recipe)
nnbake <- bake(nnprep, train)

# Define the Neural Network Model
nn_model <- mlp(hidden_units = tune(), epochs = 50) %>%
  set_engine("keras", verbose = 0) %>%  # Set engine with verbosity to reduce console output
  set_mode("classification")

# Set up the Workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

# Define Tuning Grid for `hidden_units`
nn_tuneGrid <- grid_regular(hidden_units(range = c(1L, 15L)), levels = 4)

# Cross-validation
folds <- vfold_cv(train, v = 5, repeats = 1)

# Tune the Neural Network Model
tuned_nn <- nn_wf %>%
  tune_grid(
    resamples = folds,
    grid = nn_tuneGrid,
    metrics = metric_set(accuracy)  # Use accuracy metric for tuning
  )

# Plot the tuning results
tuned_nn %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() +
  labs(title = "Tuning Neural Network: Accuracy vs Hidden Units")

# Find Best Tuning Parameters
nnbestTune <- tuned_nn %>%
  select_best()

# Finalize the Workflow and Fit it
nn_final_wf <- nn_wf %>%
  finalize_workflow(nnbestTune) %>%
  fit(data = train)

# Predict on the Test Set
nnfinal <- nn_final_wf %>%
  predict(new_data = test, type = "class")



nn_submission <- nnfinal %>%
  bind_cols(.,test) %>%
  select(id, .pred_class)%>%
  rename(type=.pred_class) # Select 'id' and 'type' columns for the submission

vroom_write(x=nn_submission, file = "nn.csv",delim = ",")
###Boosting and Bart

library(bonsai)
library(lightgbm)
library(dbarts)

boost_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%  # Turn color to factor and dummy encode
  step_range(all_numeric_predictors(), min = 0, max = 1)  # Scale to [0,1]

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")


## Naive Bayes
library(tidymodels)
library(discrim)
library(naivebayes)

nb_recipe <- nn_recipe <- recipe(type ~ ., data = train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_normalize(all_numeric_predictors())

nbprep <- prep(nb_recipe)
nbbake <- bake(nbprep, train)

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
add_recipe(nb_recipe) %>%
add_model(nb_model)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness())
folds_nb <- vfold_cv(train, v = 4, repeats = 1)

nb_cv_results <- nb_wf %>%
  tune_grid(resamples = folds_nb,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc))

besttune_nb <- nb_wf %>%
  select_best(metric = 'roc_auc')

# Finalize the Workflow and Fit it
nb_final_wf <- nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data = train)

# Predict on the Test Set
nbfinal <- nb_final_wf %>%
  predict(new_data = test, type = "class")



nb_submission <- nbfinal %>%
  bind_cols(.,test) %>%
  select(id, .pred_class)%>%
  rename(type=.pred_class) # Select 'id' and 'type' columns for the submission

vroom_write(x=nn_submission, file = "nb.csv",delim = ",")