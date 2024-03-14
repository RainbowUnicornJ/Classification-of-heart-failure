install.packages("skimr")
install.packages("ggplot2")
install.packages("rsample")
install.packages("mlr3")
install.packages("mlr3verse")
install.packages("data.table")
install.packages("ranger")
install.packages("xgboost")
install.packages("mlr3tuning")
install.packages("pROC")
install.packages("keras")
install_keras()
library(skimr)
library(ggplot2)
library(rsample)
library(mlr3)
library(mlr3verse)
library(data.table)
library(ranger)
library(xgboost)
library(mlr3tuning)
library(pROC)
library(keras)

# loaded the data about heart failure
data <- read.csv("heart_failure.csv")

# check the data information
summary(data)
str(data)
skim(data)

# check if there are missing values
sum(is.na(data))

# convert categorical variables into factors
data$fatal_mi <- as.factor(data$fatal_mi)

# try to plot some graphs about the relationship with age, ejection fraction and fatal mi
ggplot(data, aes(x=age)) + geom_histogram(binwidth = 10, fill="grey", color = "black")
ggplot(data, aes(x=fatal_mi)) + geom_bar(fill="grey",color = "black") + ggtitle("Distribution of Fatal mi")
ggplot(data, aes(x=fatal_mi, y=age)) + geom_boxplot() + ggtitle("Age versus Fatal mi")
ggplot(data, aes(x=fatal_mi, y=ejection_fraction)) + geom_boxplot() + ggtitle("Ejection fraction versus fatal mi")

# partition the data set 70% 15% 15%
set.seed(212)
split_data <- initial_split(data, prop=0.7)
train_data <- training(split_data)
test_val_data <- testing(split_data)
split2_data <- initial_split(test_val_data, prop=0.5)
val_data <- testing(split2_data)
test_data <- training(split2_data)

# define mlr3 task
task <- TaskClassif$new(id="fmi", backend=train_data, target = "fatal_mi")

# deal with the factor feature for xgboost to one-hot encoding
po_encode <- po("encode")

# learners
base <- lrn("classif.featureless", predict_type="prob")
log_reg <- lrn("classif.log_reg", predict_type="prob")
cart <- lrn("classif.rpart", predict_type="prob")
ranger <- lrn("classif.ranger", predict_type="prob", importance="impurity")
xgboost <- lrn("classif.xgboost", predict_type="prob")

# create a pipe conbined with one-hot encoding and learner
graph <- po_encode %>>% xgboost

# apply cross validation
set.seed(123)
res_cv <- rsmp("cv", folds=10)
res_cv$instantiate(task)

# resampling
res_base <- resample(task, base, res_cv, store_models = TRUE)
res_log_reg <- resample(task, log_reg, res_cv, store_models = TRUE)
res_cart <- resample(task, cart, res_cv, store_models = TRUE)
res_ranger <- resample(task, ranger, res_cv, store_models = TRUE)
res_xgboost <- resample(task, GraphLearner$new(graph), res_cv, store_models = TRUE)

# indicators of aggregation
measures <- list(msr("classif.ce"), msr("classif.acc"), msr("classif.fpr"), msr("classif.fnr"), msr("classif.auc"))
agg_base <- res_base$aggregate(measures)
agg_log_reg <- res_log_reg$aggregate(measures)
agg_cart <- res_cart$aggregate(measures)
agg_ranger <- res_ranger$aggregate(measures)
agg_xgboost <- res_xgboost$aggregate(measures)

agg_base
agg_log_reg
agg_cart
agg_ranger
agg_xgboost

# plot the data with indicators and models
all_models <-  data.frame(
  model = rep(c("Base","Logistic regression", "Cart", "Range", "Xgboost"), each=5),
  indicator = rep(c("Classification CE", "Classification ACC", "Classification FPR", "Classification FNR", "Classification AUC"), times=5),
  value = c(agg_base, agg_log_reg, agg_cart, agg_ranger, agg_xgboost)
)

ggplot(all_models, aes(x=model, y=value, fill=indicator)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x="Model", y="Value", title="Model performance comparison", fill="Indicator")

# hyperparameter selection and set the range of parameters
hyperpara <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 100, upper = 1000),
  ParamInt$new("max.depth", lower = 1, upper = 20),
  ParamInt$new("min.node.size", lower = 1, upper = 20)
))

set.seed(123)
# tuner using random search
tuner <- AutoTuner$new(
  learner = ranger,
  resampling = res_cv,
  measure = msr("classif.auc"),
  search_space = hyperpara,
  terminator = trm("evals", n_evals=100),
  tuner = tnr("random_search")
)

# train the tuner and get the best result 
tuner$train(task)

# got the combination of the best results
best_result <- tuner$archive$best()
best_para <- best_result$x
print(best_para)

# use the best hyperparameter to create a learner about random forest
best_ranger <- lrn("classif.ranger", predict_type = "prob", importance="impurity", num.trees = best_result$num.trees, 
                   max.depth = best_result$max.depth, min.node.size = best_result$min.node.size)

# training
best_ranger$train(task)

# define a task using test_data
test_task <- TaskClassif$new(id = "fmi_test", backend = test_data, target = "fatal_mi")

# predict
pred <- best_ranger$predict(test_task)

# calculate the auc
auc <- pred$score(msr("classif.auc"))
auc

# extract the importance form the best ranger model
imp_features <- best_ranger$model$variable.importance

# create a data frame and order the features by the importance
df_features <- data.frame(Feature = names(imp_features), Importance = imp_features)
df_features <- df_features[order(-df_features$Importance), ]

# print the result
print(df_features)

# plot the bar chart about the importance of feature
ggplot(df_features, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  labs(x="Feature", y="Importance", title="A bar chart about the importance of feature")

# plot the ROC curve
label <- test_task$truth() #real label 
prob <- pred$prob[, 2] # probability of prediction with fatal mi

roc <- roc(response = label, predictor = prob)

ggplot(data = data.frame(tpr = roc$sensitivities, fpr = roc$specificities), aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = paste("ROC Curve of random forest (AUC =", round(auc, 3), ")"),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) 

# define a model
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = c(ncol(train_data)-1)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# complie model and specify the loss function, etc.
deep.net %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

summary(deep.net)

# separate labels and features
train_label <- as.numeric(as.character(train_data$fatal_mi))
train_feature <- train_data[, -ncol(train_data)]  # remove the column except the last
val_label <- as.numeric(as.character(val_data$fatal_mi))
val_feature <- val_data[, -ncol(val_data)]
test_label <- as.numeric(as.character(test_data$fatal_mi))
test_feature <- test_data[, -ncol(test_data)] # remove the column except the last

x_train <- as.matrix(train_feature)
x_val <- as.matrix(val_feature)
y_train <- train_label
y_val <- val_label
test_feature <- as.matrix(test_feature)

# train the model
set.seed(200)
deep.net %>% fit(
  x = x_train, y = y_train,
  epochs = 50, batch_size = 32,
  validation_data = list(x_val, y_val)
)

# assess the performance
score <- deep.net %>% evaluate(test_feature, test_label, verbose = 0)
cat('Test loss:', score[1], '\n')
cat('Test accuracy:', score[2], '\n')

# prediction
prediction_deep <- deep.net %>% predict(test_feature)
prob_pred <- prediction_deep[, 1]

# ROC curve
roc_test <- roc(test_label, prob_pred)
plot(roc_test)

