import titanic_src.features.transform_features as tf
import titanic_src.utils

from sklearn.preprocessing import CategoricalEncoder, QuantileTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import MICEImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

train = titanic_src.utils.read_csv('data/interim/added_features_train.csv', index_col=0)

y = train['Survived'].as_matrix()

cat_nominal_cols = ['Title', 'IsAlone', 'Sex', 'Embarked', 'HighestDeck']
cat_ordinal_cols = ['Pclass']
numerical_cols = ['Age', 'SibSp', 'Fare', 'FamilySize', 'NumOfCabins']

cat_nominal_pipeline = Pipeline([
    ('selector', tf.DataFrameSelector(cat_nominal_cols)),
    ('impute', tf.CustomImputer(strategy='mode')),
    ('encoder', CategoricalEncoder(encoding='onehot-dense')),
])

cat_ordinal_pipeline = Pipeline([
    ('selector', tf.DataFrameSelector(cat_ordinal_cols)),
    ('impute', tf.CustomImputer(strategy='mode')),
    ('encode_scale', QuantileTransformer(output_distribution='normal'))
])

num_init_quantile_transformer = QuantileTransformer(output_distribution='normal')

def inverse_func(X):
    return num_init_quantile_transformer.inverse_transform(X)


numerical_pipeline = Pipeline([
    ('selector', tf.DataFrameSelector(numerical_cols)),
    ('scale', num_init_quantile_transformer),
])

combined_features = FeatureUnion([
    ('numerical_pipeline', numerical_pipeline),
    ('cat_nominal_pipeline', cat_nominal_pipeline),
    ('cat_ordinal_pipeline', cat_ordinal_pipeline),
])

mice_pipeline = Pipeline([
    ('combined_features', combined_features),
    ('mice_impute', MICEImputer()),
    ('reverse_quantile_transform',
     tf.SelectiveAction(col=list(range(len(numerical_cols))), action=FunctionTransformer()))
])

feature_transform_pipeline = Pipeline([
    ('mice_pipeline', mice_pipeline),
    ('inverse_qt', tf.SelectiveAction(col=list(range(len(numerical_cols))),
                                      action=FunctionTransformer(inverse_func))),
    ('feature_scaling', tf.SelectiveAction(col=(range(len(numerical_cols))),
                                           action=QuantileTransformer(output_distribution='normal'))),
    ('feature_selection', None),
    ('model', RandomForestClassifier())
])

cat_ordinal_pipeline_encode_scale = [CategoricalEncoder(encoding='onehot-dense'),
                                     QuantileTransformer(output_distribution='normal'),
                                     StandardScaler()]
feature_selection_options = [PCA(n_components=5), PCA(n_components=10), PCA(n_components=20), None]
numerical_pipeline_scale = [QuantileTransformer(output_distribution='normal'), StandardScaler(), RobustScaler()]

# In[17]:


from sklearn.model_selection import GridSearchCV

# In[23]:


feature_transform_pipeline.fit(train, y)



params = {#'feature_scaling__action': numerical_pipeline_scale,
          'feature_selection': feature_selection_options,
          #'mice_pipeline__combined_features__cat_ordinal_pipeline__encode_scale':cat_ordinal_pipeline_encode_scale
          }

gridsearch = GridSearchCV(feature_transform_pipeline, params, scoring='accuracy', cv=5).fit(train, y)

#
# # In[236]:
#
#
# # X = feature_transform_pipeline.fit_transform(train)
#
#
# # In[234]:
#
#
# X.shape
#
#
# # In[235]:
#
#
# for each in sorted(feature_transform_pipeline.get_params().keys()):
#     print(each)
#
#
# # In[187]:
#
#
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import SGDClassifier
#
# sgd_clf = SGDClassifier(random_state=42,max_iter=100,tol=1e-3)
# # sgd_clf.fit(X, y)
#
#
# cross_val_score(sgd_clf, X, y, cv=10, scoring="accuracy")
#
#
# # In[200]:
#
#
# from sklearn.tree import DecisionTreeClassifier
#
# dtc = DecisionTreeClassifier()
#
# cross_val_score(dtc, X, y, cv=10, scoring="accuracy")
#
#
# # In[15]:
#
#
# from sklearn.ensemble import RandomForestClassifier
#
#
# # In[207]:
#
#
# rnd_clf = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=16, n_jobs=-1)
#
# cross_val_score(rnd_clf, X, y, cv=10, scoring="accuracy")
#
#
# # In[210]:
#
#
# rnd_clf.fit(X,y)
#
#
# # In[211]:
#
#
# rnd_clf.feature_importances_
#
#
# # In[206]:
#
#
# from sklearn.ensemble import GradientBoostingClassifier
#
# gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=1000, learning_rate=1.0)
#
# cross_val_score(gbrt, X, y, cv=10, scoring="accuracy")
#
#
# # In[213]:
#
#
# from sklearn.ensemble import AdaBoostClassifier
#
# ada_clf = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=1), n_estimators=200,
#     algorithm="SAMME.R", learning_rate=0.5)
# cross_val_score(ada_clf, X, y, cv=10, scoring="accuracy")
#
#
# # In[215]:
#
#
# from sklearn.neighbors import KNeighborsClassifier
#
# knc = KNeighborsClassifier()
#
# cross_val_score(knc, X, y, cv=10, scoring="accuracy")
