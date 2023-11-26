from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Variables de interes sacadas del Analisis Exploratorio
# de Datos.
INTEREST_VARIABLES = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "CODE_GENDER",
    "OWN_CAR_AGE"
]

def model_pipeline(X, y, model):
    """Funcion para hacer preprocesamiento y entrenamiento
    del modelo.

    Parameters
    ----------
    X : DataFrame
        Conjunto de datos con las variables de interes ya filtradas.
    y : Serie
        Variable a predecir.
    model : Estimador
        Modelo a entrenar.

    Returns
    -------
    Estimador
        Modelo ya entrenado.
    """
    
    # Identificar las variables categoricas y numericas
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype in ['object']]

    # Preprocesamiento para las variables numericas
    numerical_transformer = SimpleImputer(strategy="constant", fill_value=0)

    # Preprocesamiento para las variables categoricas
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Pipeline de procesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    # Pipeline de procesamiento y entrenamiento
    pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
    ])

    # Fit the model
    pipe.fit(X, y)

    return pipe