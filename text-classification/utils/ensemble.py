import numpy as np
from sklearn.base import clone

def cost_function(y_true, y_pred, minority_label="less useful", cost_factor=2.0):
    """
    Computes the cost of misclassification, applying a higher penalty when the minority class is misclassified.

    Args:
        y_true: The true label.
        y_pred: The predicted label.
        minority_label: The class that should be penalized more when misclassified.
        cost_factor: The penalty factor for misclassifying the minority class.

    Returns:
        A float representing the misclassification cost.
    """
    # Return zero cost if the prediction is correct
    if y_true == y_pred:
        return 0.0
    # Apply higher cost when minority class is misclassified
    elif y_true == minority_label:
        return cost_factor
    # Default misclassification cost
    else:
        return 1.0

def adacost_train(X, y, base_estimator, n_estimators=50, cost_factor=2.0):
    """
    Trains a cost-sensitive ensemble using a variant of AdaBoost (AdaCost).

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Labels corresponding to each sample.
        base_estimator: A scikit-learn estimator to be used as the base learner.
        n_estimators: The maximum number of estimators to train.
        cost_factor: The penalty factor for misclassifying the minority class.

    Returns:
        A tuple (estimators, estimator_weights):
            - estimators: A list of trained estimators.
            - estimator_weights: A list of weights corresponding to each estimator.
    """
    # Initialize sample weights evenly
    n_samples = X.shape[0]
    sample_weights = np.ones(n_samples) / n_samples
    estimators = []
    estimator_weights = []
    y = np.array(y)
    
    # Iteratively train estimators and update weights
    for i in range(n_estimators):
        estimator = clone(base_estimator)
        estimator.fit(X, y, sample_weight=sample_weights)
        y_pred = estimator.predict(X)
        incorrect = (y_pred != y)
        
        # Calculate the weighted error
        error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
        if error >= 0.5:
            break
        
        # Compute estimator weight
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        estimator_weights.append(alpha)
        estimators.append(estimator)
        
        # Update sample weights using cost function
        new_weights = []
        for j in range(n_samples):
            c = cost_function(y[j], y_pred[j], minority_label="less useful", cost_factor=cost_factor)
            new_weight = sample_weights[j] * np.exp(alpha * c)
            new_weights.append(new_weight)
        
        sample_weights = np.array(new_weights)
        sample_weights /= np.sum(sample_weights)
    
    return estimators, estimator_weights

def adacost_predict(X, estimators, estimator_weights):
    """
    Predicts labels using the trained AdaCost ensemble.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        estimators: A list of trained estimators.
        estimator_weights: A list of estimator weights.

    Returns:
        An array of predicted labels.
    """
    # Accumulate weighted predictions
    pred_sum = np.zeros(X.shape[0])
    for estimator, alpha in zip(estimators, estimator_weights):
        pred = estimator.predict(X)
        pred_mapped = np.where(pred == "more useful", 1, -1)
        pred_sum += alpha * pred_mapped
    
    # Final prediction is based on the sign of weighted sum
    final_pred = np.where(pred_sum >= 0, "more useful", "less useful")
    return final_pred
