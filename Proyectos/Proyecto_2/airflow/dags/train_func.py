def train_covertype_models():
    """
    Train multiple ML models on covertype clean data and log results to MLflow.
    Adapted for covertype forest cover type prediction task.
    """    
    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")  # Docker service name
    mlflow.set_experiment("covertype_classification")
    
    # Load clean data from database
    hook = MySqlHook(mysql_conn_id="mysql_trn")
    
    with hook.get_conn() as conn:
        df = pd.read_sql("""
            SELECT elevation, aspect, slope, horizontal_distance_to_hydrology,
                   vertical_distance_to_hydrology, horizontal_distance_to_roadways,
                   hillshade_9am, hillshade_noon, hillshade_3pm,
                   horizontal_distance_to_fire_points, wilderness_area, 
                   soil_type, cover_type
            FROM covertype_clean;
        """, conn)
    
    print(f"[TRAIN] Loaded {len(df)} samples from covertype_clean")
    
    # Prepare features and target
    X = df.drop(columns=['cover_type'])
    y = df['cover_type']
    
    print(f"[TRAIN] Features shape: {X.shape}")
    print(f"[TRAIN] Target distribution:\n{y.value_counts().sort_index()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessor
    numeric_features = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
                       'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
                       'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
                       'horizontal_distance_to_fire_points']
    
    categorical_features = ['wilderness_area', 'soil_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define models to train
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svc": SVC(kernel='rbf', probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    
    metrics = []
    best_name, best_f1 = None, -1.0
    
    # Train and log each model
    for name, base_model in models.items():
        print(f"[TRAIN] Training {name}...")
        
        # Create full pipeline
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", base_model)
        ])
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"covertype_{name}"):
            # Train model
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            
            # Log basic parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(np.unique(y)))
            
            # Log model hyperparameters
            try:
                params = base_model.get_params()
                clean_params = {k: v for k, v in params.items() 
                              if isinstance(v, (int, float, str, bool, type(None)))}
                mlflow.log_params(clean_params)
            except Exception as e:
                print(f"[TRAIN] Warning: Could not log params for {name}: {e}")
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            
            # Log detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{label}_{metric_name}", value)
            
            # Log model to MLflow
            try:
                mlflow.sklearn.log_model(
                    clf, 
                    artifact_path="model",
                    registered_model_name="CovertypeClassifier"
                )
                print(f"[TRAIN] Model {name} logged to MLflow successfully")
            except Exception as e:
                print(f"[TRAIN] Warning: Could not log model {name}: {e}")
            
            # Store metrics for comparison
            metrics.append({
                "model": name, 
                "accuracy": acc, 
                "f1_macro": f1,
                "train_size": len(X_train),
                "test_size": len(X_test)
            })
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
            
            print(f"[TRAIN] {name} - Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")
    
    # Create metrics summary
    metrics_df = pd.DataFrame(metrics).sort_values("f1_macro", ascending=False)
    
    # Log summary metrics to MLflow
    with mlflow.start_run(run_name="covertype_model_comparison"):
        mlflow.log_param("experiment_type", "model_comparison")
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_f1_macro", best_f1)
        
        # Create and log metrics table as artifact
        metrics_csv = "/tmp/covertype_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        mlflow.log_artifact(metrics_csv, artifact_path="results")
    
    # Print results
    print("\n[METRICS] Model Performance Summary:")
    print(metrics_df.to_string(index=False))
    print(f"\n[BEST] {best_name} (f1_macro={best_f1:.4f})")
    
    return {
        "best_model": best_name,
        "best_f1_score": best_f1,
        "total_models_trained": len(models),
        "metrics": metrics_df.to_dict('records')
    }