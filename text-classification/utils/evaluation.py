import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import learning_curve

from lime import lime_tabular

def tune_threshold(model, X_val, y_val, positive_class_label="more useful", thresholds=np.linspace(0, 1, 50)):
    """
    Tune the decision threshold for converting predicted probabilities to binary predictions.

    Parameters:
    - model: A fitted classifier with a predict_proba method.
    - X_val: Validation feature set.
    - y_val: True labels for the validation set.
    - positive_class_label: The label that should be considered the positive class.
    - thresholds: An array of thresholds to evaluate.

    Returns:
    - best_threshold: The threshold that maximizes the macro F1 score.
    - best_metric: The best macro F1 score achieved.
    """
    best_threshold = 0.5
    best_metric = 0.0

    # Convert labels in y_val to 0 or 1, depending on whether they match the positive class.
    y_val_binary = (np.array(y_val) == positive_class_label).astype(int)
    
    # Evaluate each threshold by computing the macro F1 score on validation data.
    for t in thresholds:
        # Use predicted probabilities for the positive class (column index 1).
        y_proba = model.predict_proba(X_val)[:, 1]
        # Convert probabilities to binary predictions based on threshold t.
        y_pred = (y_proba >= t).astype(int)
        # Calculate macro F1 score for this threshold.
        _, _, f1, _ = precision_recall_fscore_support(y_val_binary, y_pred, average="macro", zero_division=0)
        # Track the threshold that yields the highest macro F1.
        if f1 > best_metric:
            best_metric = f1
            best_threshold = t
    
    return best_threshold, best_metric

def plot_learning_curve(model, X, y, cv=5, scoring="f1_weighted", n_jobs=-1, figsize=(8, 6)):
    """
    Plot a learning curve for a given estimator.

    Parameters:
    - model: The estimator (model) for which the learning curve is computed.
    - X: Feature set.
    - y: Target labels.
    - cv: Cross-validation splitting strategy.
    - scoring: Scoring metric for evaluation.
    - n_jobs: Number of jobs to run in parallel.
    - figsize: Tuple for the figure size.

    Displays a plot of the training and validation scores.
    """
    # Compute the learning curve data for both training and validation sets.
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
    )
    
    # Calculate the mean of the training and validation scores at each train size.
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    
    # Plot the mean training and validation scores as a function of the training set size.
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', label="Validation score")
    plt.xlabel("Number of Training Samples")
    plt.ylabel(scoring)
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.show()

def plot_feature_importances_from_pipeline(pipeline, X_train, feature_names=None, top_n=20):
    """
    Plots top_n feature importances mapping back from PCA space if PCA is used.

    Parameters:
    - pipeline: A scikit-learn Pipeline containing possibly a PCA step and a classifier.
    - X_train: The training features used for fitting the pipeline.
    - feature_names: Optional list of feature names. Will use X_train.columns if not provided.
    - top_n: Number of top features to display.
    """
    # Extract pipeline steps for PCA and the classifier.
    pca_step = pipeline.named_steps.get('pca', None)
    rf_clf   = pipeline.named_steps.get('clf', None)
    
    # Check if a classifier with feature_importances_ is present.
    if rf_clf is None or not hasattr(rf_clf, "feature_importances_"):
        print("No feature importances available in this pipeline.")
        return

    importances = rf_clf.feature_importances_

    # If feature names aren't provided, try to infer them from X_train. Otherwise, create placeholder names.
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = np.array([f"feature_{i}" for i in range(X_train.shape[1])])
    else:
        feature_names = np.array(feature_names)
        
    # If PCA is used, map feature importances from the transformed space back to original features.
    if pca_step is None or pca_step == 'passthrough':
        original_importances = importances
    else:
        components = pca_step.components_
        n_components, n_original_feats = components.shape
        original_importances = np.zeros(n_original_feats)
        # Accumulate absolute contributions of each principal component scaled by the classifier's importance.
        for i in range(n_components):
            original_importances += np.abs(components[i]) * importances[i]
        original_importances /= original_importances.sum()
    
    # Sort by descending importance and select top_n features.
    indices = np.argsort(original_importances)[::-1]
    sorted_importances = original_importances[indices][:top_n]
    sorted_features = feature_names[indices][:top_n]
    
    # Plot the bar chart of the top feature importances.
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), sorted_importances, align='center')
    plt.xticks(range(top_n), sorted_features, rotation=90)
    title = f"Top {top_n} Feature Importances"
    if pca_step not in [None, 'passthrough']:
        title += " (mapped from PCA)"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def comprehensive_evaluation(model, 
                             X_train_ready, y_train, 
                             X_test_ready, y_test,
                             model_type='sklearn',
                             feature_names=None,
                             cv=5,
                             scoring="f1_weighted",
                             n_jobs=-1,
                             sample_index_for_lime=0,
                             history=None,
                             original_X_train=None,
                             original_X_test=None):
    """
    Comprehensive evaluation of a model.

    For scikit-learn based models (model_type='sklearn'):
      - Plots a learning curve using the given scoring metric.
      - Prints train and test classification reports.
      - Plots confusion matrix, ROC, and precision–recall curves.
      - Plots feature importances if available.
      - Shows a LIME explanation for 10 test instances using the preprocessed data for predictions,
        and prints the corresponding original 'text' from original_X_test.

    For deep learning models (model_type in ['keras', 'bert', 'multi_input']):
      - Evaluates the model on the test set and prints a classification report.
      - If training history is provided, plots loss and accuracy curves.

    Parameters:
      model: trained model (scikit-learn estimator, Keras model, etc.)
      X_train_ready, y_train: training data (preprocessed/numeric)
      X_test_ready, y_test: test data (preprocessed/numeric)
      model_type: one of 'sklearn', 'keras', 'bert', or 'multi_input'
      feature_names: list or array of feature names (if applicable)
      cv: number of folds for the learning curve (only for scikit-learn models)
      scoring: scoring metric for the learning curve
      n_jobs: number of jobs for computing the learning curve
      sample_index_for_lime: starting index in X_test_ready for LIME explanation (for tabular sklearn models)
      history: training history for Keras models (optional)
      original_X_train, original_X_test: original data (with a 'text' column) corresponding to X_train_ready and
                                         X_test_ready; these are used solely for retrieving the original text in the LIME printout.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
    import lime
    import lime.lime_tabular

    # Handle scikit-learn models
    if model_type == 'sklearn':
        # ---- Learning Curve ----
        try:
            from sklearn.model_selection import learning_curve
            print("Plotting learning curve...")
            # Compute learning curve for training vs. validation performance.
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train_ready, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            
            # Plot the average training and validation scores across different train sizes.
            plt.figure(figsize=(8, 6))
            plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
            plt.xlabel("Training Examples")
            plt.ylabel(scoring)
            plt.title("Learning Curve")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Learning curve plotting failed:", e)
        
        # ---- Classification Reports (Train and Test) ----
        print("\nEvaluating on training data:")
        y_train_pred = model.predict(X_train_ready)
        print(classification_report(y_train, y_train_pred, digits=4))
        
        print("\nEvaluating on test data:")
        y_test_pred = model.predict(X_test_ready)
        print(classification_report(y_test, y_test_pred, digits=4))
        
        # ---- Confusion Matrix ----
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix", fontsize=16, pad=20)
        plt.ylabel("Actual", fontsize=14)
        plt.xlabel("Predicted", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.show()
        
        # ---- ROC and Precision-Recall Curves ----
        if hasattr(model, "predict_proba"):
            try:
                # Extract predicted probabilities for the positive class (label "more useful").
                y_proba = model.predict_proba(X_test_ready)[:, 1]
                # Compute FPR, TPR for ROC, then calculate the AUC score.
                fpr, tpr, _ = roc_curve((y_test == "more useful").astype(int), y_proba)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:0.2f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.show()
                
                # Compute and plot the precision-recall curve.
                precision, recall, _ = precision_recall_curve((y_test == "more useful").astype(int), y_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label="Precision-Recall Curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print("ROC/Precision-Recall curve plotting failed:", e)
        else:
            print("Model does not provide predict_proba for ROC/PR curves.")
        
        # ---- Feature Importances (if available) ----
        if hasattr(model, "named_steps"):
            if "clf" in model.named_steps and hasattr(model.named_steps["clf"], "feature_importances_"):
                try:
                    print("Plotting feature importances from pipeline...")
                    # Calls the helper function to handle pipeline logic and PCA, if any.
                    plot_feature_importances_from_pipeline(model, X_train_ready, feature_names=feature_names, top_n=20)
                except Exception as e:
                    print("Feature importance plotting failed:", e)
        elif hasattr(model, "feature_importances_"):
            # Plot feature importances for a non-pipeline model directly.
            importances = model.feature_importances_
            if feature_names is None:
                if hasattr(X_train_ready, 'columns'):
                    feature_names = X_train_ready.columns
                else:
                    feature_names = np.array([f"feature_{i}" for i in range(X_train_ready.shape[1])])
            indices = np.argsort(importances)[::-1]
            sorted_importances = importances[indices][:20]
            sorted_features = np.array(feature_names)[indices][:20]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
            plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
            plt.title("Feature Importances")
            plt.tight_layout()
            plt.show()
        else:
            print("No feature importance attribute found for this model.")
        
        # ---- LIME Explanation for 10 Test Instances ----
        try:
            # Create LIME explainer using the training data.
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_ready.values if hasattr(X_train_ready, "values") else X_train_ready,
                feature_names=feature_names if feature_names is not None else np.array([f"feature_{i}" for i in range(X_train_ready.shape[1])]),
                class_names=np.unique(y_train),
                mode='classification'
            )
            
            print("=" * 80)
            print("LIME Explanations for 10 Test Instances")
            print("=" * 80)
            
            aggregated_results = []
            
            # Analyze and store explanations for 10 instances of the test set, starting at sample_index_for_lime.
            for i in range(sample_index_for_lime, sample_index_for_lime + 10):
                # Retrieve instance from the test data.
                if hasattr(X_test_ready, "iloc"):
                    instance = X_test_ready.iloc[i]
                else:
                    instance = X_test_ready[i]
                
                # Reshape single instance into 2D array for prediction.
                instance_2d = instance.values.reshape(1, -1) if hasattr(instance, "values") else instance.reshape(1, -1)
                predicted_class = model.predict(instance_2d)[0]
                
                # Explain the instance predictions using LIME.
                exp = lime_explainer.explain_instance(
                    instance.values if hasattr(instance, "values") else instance,
                    model.predict_proba,
                    num_features=10
                )
                exp_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
                
                # If original data (with text) is provided, retrieve corresponding raw text for reference.
                if original_X_test is not None and hasattr(original_X_test, "iloc") and "text" in original_X_test.columns:
                    original_text = original_X_test.iloc[i]["text"]
                else:
                    original_text = "N/A"
                
                aggregated_results.append({
                    "Instance": i,
                    "Predicted Class": predicted_class,
                    "Original Text": original_text,
                    "LIME Explanation": exp_df
                })
            
            # Display the results for each of the 10 instances, including the original text and top features.
            for res in aggregated_results:
                print("\n" + "-" * 80)
                print(f"Instance {res['Instance']} - Predicted Class: {res['Predicted Class']}")
                print(f"Original Text:\n{res['Original Text']}")
                print("Top 10 Features and Their Contribution Scores:")
                print(res['LIME Explanation'].to_string(index=False))
                print("-" * 80)
                
        except Exception as e:
            print("LIME explanation failed:", e)
    
    # Handle deep learning models (Keras, BERT, multi-input, etc.).
    elif model_type in ['keras', 'bert', 'multi_input']:
        print("\nEvaluating deep learning model on test set...")
        # Evaluate model on test data (loss and metric scores).
        eval_results = model.evaluate(X_test_ready, y_test, verbose=0)
        print("Test evaluation results:", eval_results)
        
        # Make predictions and convert probabilities/logits to discrete labels.
        y_proba = model.predict(X_test_ready)
        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = (y_proba > 0.5).astype(int).flatten()
        
        # Print a classification report with standard metrics.
        from sklearn.metrics import classification_report
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred, digits=4))
        
        # If history is available, plot the training loss and (if present) accuracy curves.
        if history is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['loss'], label="Train Loss")
            plt.plot(history.history.get('val_loss', []), label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            if 'accuracy' in history.history:
                plt.figure(figsize=(8, 6))
                plt.plot(history.history['accuracy'], label="Train Accuracy")
                plt.plot(history.history.get('val_accuracy', []), label="Validation Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("Training Accuracy Curve")
                plt.legend()
                plt.tight_layout()
                plt.show()
    else:
        print("Unknown model type. Please choose 'sklearn', 'keras', 'bert', or 'multi_input'.")
