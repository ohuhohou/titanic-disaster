# Titanic survival prediction - R version.
#
# Loads src/data/train.csv and src/data/test.csv, cleans and engineers features,
# fits a logistic regression (glm, binomial), reports accuracy and writes predictions.
# Every step prints what it changed so the full workflow is visible in the terminal.

suppressPackageStartupMessages(library(dplyr))
options(width = 150, dplyr.width = Inf)

# Data folder: DATA_DIR env var (set in the Dockerfile) or src/data when run from the repo root.
data_dir    <- Sys.getenv("DATA_DIR", unset = file.path("src", "data"))
train_path  <- file.path(data_dir, "train.csv")
test_path   <- file.path(data_dir, "test.csv")
labels_path <- file.path(data_dir, "test_labels.csv")   # optional, see README
output_path <- file.path(data_dir, "predictions_r.csv")

# Note: Sex is NOT a model input here. Title (Mr / Mrs / Miss / Master / Rare) already encodes
# sex (and age group), and using both makes unregularised glm nearly singular (Mr is 100% male,
# Miss/Mrs 100% female -> huge, unstable coefficients). IsAlone is dropped as it is redundant
# with FamilySize. So the R model is a slightly different, more parsimonious model than Python's.
features <- c("Pclass", "Title", "Age", "FareLog", "FamilySize", "HasCabin", "Embarked")

header <- function(text) {
  cat("\n", strrep("=", 70), "\n", text, "\n", strrep("=", 70), "\n", sep = "")
}

load_csv <- function(path, name) {
  if (!file.exists(path)) {
    stop(sprintf("%s not found at %s. Download it from Kaggle into src/data/ (see README).",
                 name, path), call. = FALSE)
  }
  df <- read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
  cat(sprintf("Loaded %s: %d rows x %d columns\n", name, nrow(df), ncol(df)))
  df
}

extract_title <- function(names) {
  title <- trimws(sub("^.*?,\\s*([^\\.]+)\\..*$", "\\1", names, perl = TRUE))
  title[title %in% c("Mlle", "Ms")] <- "Miss"
  title[title == "Mme"] <- "Mrs"
  title[!title %in% c("Mr", "Mrs", "Miss", "Master")] <- "Rare"
  title
}

accuracy <- function(truth, pred) mean(truth == pred)

# Learn imputation values from the TRAINING data only (no leakage into test).
fit_cleaning_params <- function(train) {
  age_by_title <- train %>%
    mutate(Title = extract_title(Name)) %>%
    group_by(Title) %>%
    summarise(med = median(Age, na.rm = TRUE), .groups = "drop")
  list(
    age_by_title  = setNames(age_by_title$med, age_by_title$Title),
    age_overall   = median(train$Age, na.rm = TRUE),
    fare_median   = median(train$Fare, na.rm = TRUE),
    embarked_mode = names(which.max(table(train$Embarked)))
  )
}

prepare <- function(df, params, name) {
  header(sprintf("Q14 - Adjusting the %s data", name))
  miss <- colSums(is.na(df))
  cat(sprintf("Missing values before cleaning (%s):\n", name))
  print(miss[miss > 0])

  df$Title <- extract_title(df$Name)
  cat("\n[+] Added 'Title' from Name. Counts:\n")
  print(table(df$Title))

  n <- sum(is.na(df$Age))
  idx <- is.na(df$Age)
  df$Age[idx] <- params$age_by_title[df$Title[idx]]
  df$Age[is.na(df$Age)] <- params$age_overall
  cat(sprintf("\n[~] Filled %d missing Age values with the training median age for each Title (%s)\n",
              n, paste(names(params$age_by_title), params$age_by_title, sep = "=", collapse = ", ")))

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- params$fare_median
  cat(sprintf("[~] Filled %d missing Fare values with training median %.2f\n", n, params$fare_median))

  n <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- params$embarked_mode
  cat(sprintf("[~] Filled %d missing Embarked values with training mode '%s'\n", n, params$embarked_mode))

  df <- df %>%
    mutate(HasCabin   = as.integer(!is.na(Cabin)),
           FamilySize = SibSp + Parch + 1L,
           IsAlone    = as.integer(FamilySize == 1),
           FareLog    = log1p(Fare),
           Sex        = factor(Sex, levels = c("female", "male")),
           Embarked   = factor(Embarked, levels = c("C", "Q", "S")),
           Title      = factor(Title, levels = c("Mr", "Miss", "Mrs", "Master", "Rare")))
  cat(sprintf("[+] Added 'HasCabin' (Cabin is %.0f%% missing). Share with cabin: %.2f\n",
              100 * mean(is.na(df$Cabin)), mean(df$HasCabin)))
  cat(sprintf("[+] Added 'FamilySize' = SibSp + Parch + 1 (mean %.2f) and 'IsAlone' (share alone %.2f)\n",
              mean(df$FamilySize), mean(df$IsAlone)))
  cat("[+] Added 'FareLog' = log(1 + Fare) to reduce the heavy right skew of Fare\n")
  cat("[+] Converted Sex, Embarked, Title to factors (glm creates the dummy variables;",
      "baselines female / C / Mr)\n")
  cat("[-] Not used as model inputs: Name, Ticket, Cabin, SibSp, Parch, Fare",
      "(replaced by engineered columns or free-text IDs), plus Sex and IsAlone",
      "(redundant with Title and FamilySize - avoids collinearity in glm)\n")
  cat(sprintf("Final feature set (%d): %s\n", length(features), paste(features, collapse = ", ")))
  cat(sprintf("Missing values remaining in features: %d\n", sum(is.na(df[, features]))))
  df
}

model_formula <- as.formula(paste("Survived ~", paste(features, collapse = " + ")))
predict_class <- function(model, newdata) as.integer(predict(model, newdata, type = "response") >= 0.5)

# ---------------------------------------------------------------- Q13
header("Q13 - Loading training data")
train_raw <- load_csv(train_path, "train.csv")
print(head(train_raw, 5))
cat(sprintf("\nOverall survival rate in train: %.3f\n", mean(train_raw$Survived)))
cat("Survival rate by Sex:\n");    print(round(tapply(train_raw$Survived, train_raw$Sex, mean), 3))
cat("Survival rate by Pclass:\n"); print(round(tapply(train_raw$Survived, train_raw$Pclass, mean), 3))

# ---------------------------------------------------------------- Q14
params <- fit_cleaning_params(train_raw)
train  <- prepare(train_raw, params, "training")

# ---------------------------------------------------------------- Q15
header("Q15 - Logistic regression model")
# Stratified 80/20 hold-out split for an honest out-of-sample accuracy estimate.
set.seed(42)
val_idx <- unlist(lapply(split(seq_len(nrow(train)), train$Survived),
                         function(ix) sample(ix, round(0.2 * length(ix)))))
fit_80  <- glm(model_formula, data = train[-val_idx, ], family = binomial)
val_acc <- accuracy(train$Survived[val_idx], predict_class(fit_80, train[val_idx, ]))

# Final model on all training rows.
model <- glm(model_formula, data = train, family = binomial)
cat("Model: glm(", deparse(model_formula, width.cutoff = 200), ", family = binomial)\n")
cat("Coefficients (log-odds; positive = higher survival odds):\n")
print(round(coef(summary(model))[, c("Estimate", "Pr(>|z|)")], 4))

# ---------------------------------------------------------------- Q16
header("Q16 - Accuracy on the training set")
train_acc <- accuracy(train$Survived, predict_class(model, train))
cat(sprintf("Training accuracy (fit and scored on all %d rows): %.4f\n", nrow(train), train_acc))
cat(sprintf("Hold-out validation accuracy (fit on 80%%, scored on unseen 20%%): %.4f\n", val_acc))

# ---------------------------------------------------------------- Q17
header("Q17 - Predicting on the test set")
test_raw <- load_csv(test_path, "test.csv")
test     <- prepare(test_raw, params, "test")
header("Q17 - Test predictions")
test$Survived <- predict_class(model, test)
write.csv(test[, c("PassengerId", "Survived")], output_path, row.names = FALSE)
cat(sprintf("Predicted survivors: %d of %d (%.3f)\n", sum(test$Survived), nrow(test), mean(test$Survived)))
cat(sprintf("Saved predictions to %s\n", output_path))
print(head(test[, c("PassengerId", "Name", "Sex", "Pclass", "Survived")], 10))

# ---------------------------------------------------------------- Q18
header("Q18 - Accuracy on the test set")
if (file.exists(labels_path)) {
  labels <- read.csv(labels_path)[, c("PassengerId", "Survived")]
  merged <- merge(test[, c("PassengerId", "Survived")], labels,
                  by = "PassengerId", suffixes = c("_pred", "_true"))
  cat(sprintf("Test accuracy against test_labels.csv (%d rows): %.4f\n",
              nrow(merged), accuracy(merged$Survived_true, merged$Survived_pred)))
} else {
  cat("Kaggle's test.csv has NO 'Survived' column - the true labels are hidden, so test",
      "accuracy cannot be computed locally.\n")
  cat(sprintf("Best local estimate of test accuracy = hold-out validation accuracy: %.4f\n", val_acc))
  cat("To get the real test accuracy, upload predictions_r.csv to the Kaggle competition,",
      "or place a labelled test_labels.csv (PassengerId,Survived) in src/data/.\n")
}

header("Done")
