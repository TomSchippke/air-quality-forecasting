'''
This file contains baseline models for the air quality forecasting task.
They will be used to compare against the more sophisticated models in this project.
'''

import pandas as pd 
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_random_forest(
    X_test_train: pd.DataFrame, 
    y_test_train: pd.DataFrame, 
    random_state: int = 42,
    recall: bool = False,
    model_path: str = "results/models/random_forest_best.joblib",
    param_grid: dict[str, list[int]] = {
        'n_estimators': [100, 200, 500],
        'max_depth': [15, 25, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1.0, 'sqrt']
    }
) -> RandomForestRegressor:
    
    if recall and os.path.exists(model_path):
        print(f"=========================================")
        print(f"Model already existing found here : {model_path}...")
        print(f"=========================================")
        return joblib.load(model_path)
    
    print(f"=========================================")
    print(f"Starting random forest grid search...")
    print(f"=========================================")

    gs = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=random_state,
            criterion="squared_error"
        ),
        param_grid=param_grid,
        cv=5,
        refit = True,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2
    )
        
    gs.fit(X_test_train, y_test_train)
    best_rf = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    
    results_df = pd.DataFrame(gs.cv_results_)
    results_df = results_df.sort_values(by='rank_test_score')
    results_df.to_csv("results/random_forest_grid_search_results.csv", index=False)

    print(f"=========================================")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.3f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_rf, model_path)
    print(f"Best model saved in : {model_path}")
    
    print(f"=========================================")
    print(f"END of Random Forest training.")
    print(f"=========================================")

    return best_rf
    
    
